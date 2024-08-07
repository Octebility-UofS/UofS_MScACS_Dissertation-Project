from datetime import datetime
import os
import sys
import hydra

JOB_ID = "0"
if __name__ == "__main__":
    if len(sys.argv) > 1:
        JOB_ID = sys.argv.pop(1)
        sys.argv.append(f'hydra.run.dir=out/{JOB_ID}/hydra')
ROOT_DIR = os.path.join('.', 'out', JOB_ID)
os.makedirs(ROOT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import numpy as np
import jax
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
        return runner_state, transition

    return _env_step

def _make_update_step(config, env):
    def _update_step(runner_state, _):
        # Collect trajectories for replay buffer
        runner_state, traj_buffer = jax.lax.scan(
            _make_env_step(config, env),
            runner_state, None,
            length=config["REPLAY_ENV_STEPS"]
        )

        # Perform TD3 update
        train_states, env_state, last_obs, update_count, rng = runner_state
        rng, _rng = jax.random.split(rng)
        train_states, metric = make_td3_update(config)(_rng, train_states, traj_buffer)

        runner_state = (train_states, env_state, last_obs, update_count, rng)
        return runner_state, metric
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

    start_time = datetime.now()
    res = train_jit(rng)
    stop_time = datetime.now()
    print(f"Training Elapsed {stop_time-start_time}")

    metrics_y_actor_loss = np.array(res['metrics']['actor_loss'])
    # get how many non-nan values
    y_actor_loss = np.empty((metrics_y_actor_loss.shape[0], ))
    for i in range(y_actor_loss.shape[0]):
        non_nan = metrics_y_actor_loss[i][~np.isnan(metrics_y_actor_loss[i])]
        y_actor_loss[i] = np.mean(non_nan)

    y_critic_loss = np.mean(res['metrics']['critic_loss'], axis=1)

    plt.plot(np.arange(y_critic_loss.shape[0]), y_critic_loss, label="Critic Loss")
    plt.plot(np.arange(y_actor_loss.shape[0]), y_actor_loss, label="Actor Loss")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "train_loss.png"))
    plt.close()
    
    return
    rng = jax.random.PRNGKey(0)

    env = jaxmarl.make("ant_4x2")
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)

    state_history = [state.pipeline_state, ]
    for i in range(5):
        print(f"")
        actions = {}
        for agent_id in env.agents:
            rng, _rng = jax.random.split(rng)
            actions[agent_id] = env.action_space(agent_id).sample(_rng)
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, actions)
        print(info)
        return
        state_history.append(state.pipeline_state)

    # print("State", state_history[0].__dict__.keys())
    # print("Pipeline State", state_history[0].pipeline_state.__dict__.keys())
    rendered_html = html.render(env.sys, state_history)
    with open("test_html", 'w') as f:
        f.write(rendered_html)
    # print(html.render(env.sys, [s.pipeline_state.q for s in state_history]))

if __name__ == "__main__":
    main()