import os
import sys
def sys_argv_swallow(key):
    lst_arg = [ (ix, arg) for ix, arg in enumerate(sys.argv) if arg.startswith(f"{key}=") ]
    if len(lst_arg) == 0:
        return False
    elif len(lst_arg) == 1:
        ix, arg = lst_arg[0]
        sys.argv.pop(ix)
        return arg.replace(f"{key}=", "")
    else:
        raise ValueError(f"You're only allowed to supply one argument with key {key}")
   

import __main__
from datetime import datetime

from util.util import LinePlot, file_write, pickle_dump
__script_name = ".".join(os.path.split(__main__.__file__)[1].split(".")[:-1])
__time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT_DIR = os.path.join('.', 'out', 'tmp', f"0_{__time}_{__script_name}")
if __name__ == "__main__":
    out_path = sys_argv_swallow("OUT_PATH")
    job_id = sys_argv_swallow("JOB_ID")
    resume_id = sys_argv_swallow("RESUME")

    if not out_path and not job_id:
        raise ValueError("You must specify both OUT_PATH and JOB_ID, not only one")
    
    ROOT_DIR = os.path.join('.', 'out', out_path, f"{job_id}")
    sys.argv.append(f'hydra.run.dir={ROOT_DIR}/hydra')
os.makedirs(ROOT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


import hydra
import numpy as np
import jax
print(jax.devices())
import jax.numpy as jnp
from brax.io import html
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from td3.td3 import Transition, make_td3, make_td3_update

def batchify(x: dict, agent_list: list[str], num_actors: int):
    x = jnp.stack([ x[a] for a in agent_list ])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list: list[str], num_envs: int, num_actors: int):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def _make_env_step(config, env):
    def _env_step(runner_state, _):
        train_states, env_state, last_obs, update_count, rng = runner_state
        state_actor, state_actor_target, state_critic, state_critic_target = train_states

        obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

        # Select action
        rng, _rng = jax.random.split(rng)
        pi = state_actor.apply_fn(state_actor.params, obs_batch)
        action = pi.sample(seed=_rng)
        # log_prob = pi.log_prob(action) # Probably not needed, maybe only for IPPO
        env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        
        # Step environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            rng_step, env_state, env_act
        )

        transition = Transition(
            batchify(done, env.agents, config["NUM_ACTORS"]),
            action,
            batchify(reward, env.agents, config["NUM_ACTORS"]),
            obs_batch,
            batchify(obsv, env.agents, config["NUM_ACTORS"])
        )

        runner_state = (train_states, env_state, obsv, update_count, rng)
        return runner_state, (transition, reward)

    return _env_step

def _make_update_step(config, env):
    def _update_step(runner_state, _):
        # Collect trajectories for replay buffer
        runner_state, (traj_buffer, rewards) = jax.lax.scan(
            jax.jit(_make_env_step(config, env), device=jax.devices("cpu")[0]),
            runner_state, None,
            length=config["REPLAY_ENV_STEPS"]
        )

        mean_reward = jax.tree.map(
            lambda x: jnp.mean(jnp.sum(x, axis=0)),
            rewards
        )

        # Perform TD3 update
        train_states, env_state, last_obs, update_count, rng = runner_state
        rng, _rng = jax.random.split(rng)
        train_states, loss_info = make_td3_update(config)(_rng, train_states, traj_buffer)

        runner_state = (train_states, env_state, last_obs, update_count, rng)
        return runner_state, {"loss": loss_info, "reward": mean_reward}
    return _update_step

def make_train(config, rng_init):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    assert (
        (int(config["REPLAY_ENV_STEPS"]) * int(config["NUM_ACTORS"])) % int(config["BATCH_SIZE"]) == 0
    ), "Batch size must be divisible by (REPLAY_ENV_STEPS * NUM_ENVS * NUM_AGENTS)"
    config["NUM_BATCHES"] = (int(config["REPLAY_ENV_STEPS"]) * int(config["NUM_ACTORS"])) // int(config["BATCH_SIZE"])
    # Use log scale for potentiall better training results
    env = LogWrapper(env, replace_info=True)

    # Initialize TD3
    # We are assuming that the continuous action space is symmetric
    config["MAX_ACTION"] = env.action_spaces[env.agents[0]].high
    # rng_init, _rng = jax.random.split(rng_init)
    state_actor, state_actor_target, state_critic, state_critic_target = make_td3(rng_init, config, env)
    train_states = (
        state_actor,
        state_actor_target,
        state_critic,
        state_critic_target
    )

    def _train(rng: jax.dtypes.prng_key):
        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # Run train loop
        rng, _rng = jax.random.split(rng)
        update_count = 0
        runner_state = (train_states, env_state, obsv, update_count, _rng)
        runner_state, metric = jax.lax.scan(
            _make_update_step(config, env),
            runner_state, None,
            length=config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metric}
    return _train

@hydra.main(version_base=None, config_path="config", config_name="test_td3")
def main(config):
    config = OmegaConf.to_container(config)
    rng = jax.random.PRNGKey(config["JAX_SEED"])
    rng, _rng = jax.random.split(rng)

    start_time = datetime.now()
    train_jit = jax.jit(make_train(config, _rng))
    stop_time = datetime.now()
    print(f"Jit Elapsed {stop_time-start_time}")
    file_write(
        os.path.join(ROOT_DIR, 'timing.csv'),
        f"Jit Time, {stop_time-start_time}\n",
        append=True
    )

    start_time = datetime.now()
    res = train_jit(rng)
    stop_time = datetime.now()
    print(f"Training Elapsed {stop_time-start_time}")
    file_write(
        os.path.join(ROOT_DIR, 'timing.csv'),
        f"Run Time, {stop_time-start_time}\n",
        append=True
    )


    (
        state_actor,
        state_actor_target,
        state_critic,
        state_critic_target
    ) = res["runner_state"][0]
    pickle_dump(
        os.path.join(CHECKPOINT_DIR, 'final_state_actor_params.ckpt'),
        state_actor.params
    )
    pickle_dump(
        os.path.join(CHECKPOINT_DIR, 'final_state_actor_target_params.ckpt'),
        state_actor_target.params
    )
    pickle_dump(
        os.path.join(CHECKPOINT_DIR, 'final_state_critic_params.ckpt'),
        state_critic.params
    )
    pickle_dump(
        os.path.join(CHECKPOINT_DIR, 'final_state_critic_target_params.ckpt'),
        state_critic_target.params
    )
        

    metrics_y_actor_loss = np.array(res['metrics']['loss']['actor_loss'])
    # get how many non-nan values
    y_actor_loss = np.empty((metrics_y_actor_loss.shape[0], ))
    for i in range(y_actor_loss.shape[0]):
        non_nan = metrics_y_actor_loss[i][~np.isnan(metrics_y_actor_loss[i])]
        y_actor_loss[i] = np.mean(non_nan)

    y_critic_loss = np.mean(res['metrics']['loss']['critic_loss'], axis=1)

    loss_plot = LinePlot("Update Step", "Loss")
    loss_plot.add(np.arange(y_critic_loss.shape[0]), y_critic_loss, label="Critic Loss")
    loss_plot.add(np.arange(y_actor_loss.shape[0]), y_actor_loss, label="Actor Loss")
    loss_plot.save(os.path.join(ROOT_DIR, "train_loss.png"))

    pickle_dump(
        os.path.join(DATA_DIR, 'mean-reward.pkl'),
        res['metrics']['reward']
    )

    reward_plot = LinePlot("Update Step", "Mean Reward")
    reward_data = res['metrics']['reward']['__all__']
    reward_plot.add(np.arange(reward_data.shape[0]), reward_data)
    reward_plot.save(os.path.join(ROOT_DIR, 'cumulative-mean-reward.png'))

if __name__ == "__main__":
    main()