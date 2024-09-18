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

if __name__ == "__main__":
    backend_arg = sys_argv_swallow("BACKEND")
    if backend_arg == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

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

    if (out_path or job_id):
        if not (out_path and job_id):
            raise ValueError("You must specify both OUT_PATH and JOB_ID, not only one")

        ROOT_DIR = os.path.join('.', 'out', out_path, f"{job_id}")
    sys.argv.append(f'hydra.run.dir={ROOT_DIR}/hydra')
os.makedirs(ROOT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

print(ROOT_DIR)


import hydra
import numpy as np
import jax
import jax.experimental
print(jax.devices())
import jax.numpy as jnp
from brax.io import html
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from td3.td3 import TrainStates, Transition, batchify, make_td3, make_td3_update, unbatchify

def _save_checkpoints(checkpoint_steps, counter, total_num_updates, train_states):
    if counter in checkpoint_steps:
        _save_format_step = len(str(int(total_num_updates-1)))
        str_step = str(int(counter)).zfill(_save_format_step)
        state_actor, state_actor_target, state_critic, state_critic_target = train_states
        pickle_dump(
            os.path.join(CHECKPOINT_DIR, f'{str_step}_state_actor_params.ckpt'),
            state_actor.params
        )
        pickle_dump(
            os.path.join(CHECKPOINT_DIR, f'{str_step}_state_actor_target_params.ckpt'),
            state_actor_target.params
        )
        pickle_dump(
            os.path.join(CHECKPOINT_DIR, f'{str_step}_state_critic_params.ckpt'),
            state_critic.params
        )
        pickle_dump(
            os.path.join(CHECKPOINT_DIR, f'{str_step}_state_critic_target_params.ckpt'),
            state_critic_target.params
        )
    return None


def batchify_keep_num_envs(x: dict, agent_list: list[str]):
    return jnp.stack([ x[a] for a in agent_list ])


def _make_random_env_step(config, env):
    def _sample_action(rng, agent):
        arr_rng = jax.random.split(rng, config["NUM_ENVS"])
        return jax.vmap(
            env.action_space(agent).sample,
            in_axes=0
        )(arr_rng)

    def _env_step(runner_state, _):
        rng, last_env_state, last_obsv = runner_state

        # Sample random actions
        rng, _rng = jax.random.split(rng)
        arr_rng = jax.random.split(_rng, env.num_agents)
        env_action = { agent: _sample_action(arr_rng[i], agent) for i, agent in enumerate(env.agents) }

        # Step environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            rng_step, last_env_state, env_action
        )

        transition = Transition(
            batchify_keep_num_envs(done, env.agents),
            batchify_keep_num_envs(env_action, env.agents),
            batchify_keep_num_envs(reward, env.agents),
            batchify_keep_num_envs(last_obsv, env.agents),
            batchify_keep_num_envs(obsv, env.agents),
        )

        runner_state = rng, env_state, obsv
        return runner_state, (transition, reward)
    return _env_step


def _make_env_step(config, env):
    def _env_step(runner_state, _):
        rng, state_actor, last_env_state, last_obsv = runner_state

        last_obsv_batch = batchify(last_obsv, env.agents, config["NUM_ACTORS"])

        # Select Deterministic action
        action_batch = state_actor.apply_fn(state_actor.params, last_obsv_batch)
        # Add exploration noise and clip to maximum action value
        rng, _rng = jax.random.split(rng)
        action_batch = jnp.clip(
            action_batch + ( config["EXPLORATION_NOISE"] * jax.random.normal(_rng, action_batch.shape) ),
            min=-config["MAX_ACTION"], max=config["MAX_ACTION"]
        )

        env_action = unbatchify(action_batch, env.agents, config["NUM_ENVS"], env.num_agents)

        # Step environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            rng_step, last_env_state, env_action
        )

        transition = Transition(
            batchify_keep_num_envs(done, env.agents),
            batchify_keep_num_envs(env_action, env.agents),
            batchify_keep_num_envs(reward, env.agents),
            batchify_keep_num_envs(last_obsv, env.agents),
            batchify_keep_num_envs(obsv, env.agents),
        )

        runner_state = rng, state_actor, env_state, obsv
        return runner_state, (transition, reward)
    return _env_step


def make_iteration(config, env):
    def _iterate(runner_state, _):
        rng, train_states, (env_state, obsv), counter = runner_state

        # Collect trajectories equal to batch size
        rng, _rng = jax.random.split(rng)
        step_runner = _rng, train_states.state_actor, env_state, obsv
        step_runner, (trajectories, rewards) = jax.lax.scan(
            jax.jit(_make_env_step(config, env), device=jax.devices("cpu")[0]),
            step_runner, None,
            length=config["ITERATION_PARALLEL_STEPS"]
        )

        # Reshape to combine scanned dimensions and number of agents
        trajectories = jax.tree.map(
            lambda x: jnp.reshape(x, (x.shape[0]*x.shape[1], ) + x.shape[2:] ),
            trajectories
        )

        # Perform update only using the collected trajectories
        # It is not feasible to implement a replay buffer like in the TD3 PyTorch implementation
        rng, _rng = jax.random.split(rng)
        train_states, loss_info = make_td3_update(
            config
        )(_rng, train_states, trajectories, counter)

        jax.experimental.io_callback(_save_checkpoints, None, config["_CHECKPOINT_STEPS"], counter, config["NUM_ITERATIONS"], train_states)

        metrics = {"reward": rewards, "loss": loss_info}

        counter += 1
        runner_state = rng, train_states, (env_state, obsv), counter
        return runner_state, metrics
    return _iterate


def make_train(config, rng_init):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    assert(
        ( config["TOTAL_STEPS"] % config["NUM_ENVS"] ) == 0
    ), "The Total Number of Timesteps must be divisble by the Number of Environments"
    assert(
        ( config["EXPLORATION_STEPS"] % config["NUM_ENVS"] ) == 0
    ), "The Number of Exploration Steps must be divisble by the Number of Environments"
    assert(
        ( config["BATCH_SIZE"] % config["NUM_ENVS"] ) == 0
    ), "The Batch size must be divisble by the number of environments"
    # Set the number of steps that will be iterated, since we are parallelising with the environments
    _data_points_per_step = env.num_agents * int(config["NUM_ENVS"])
    config["EXPLORATION_ITERATIONS"] = int(config["EXPLORATION_STEPS"]) // int(config["NUM_ENVS"])
    _remaining_steps = int(config["TOTAL_STEPS"]) - int(config["EXPLORATION_STEPS"])
    config["ITERATION_PARALLEL_STEPS"] = int(config["BATCH_SIZE"]) // _data_points_per_step
    config["NUM_ITERATIONS"] = _remaining_steps // (config["ITERATION_PARALLEL_STEPS"] * int(config["NUM_ENVS"]))

    config["_CHECKPOINT_STEPS"] = list(np.linspace(
        0,
        config["NUM_ITERATIONS"] - 1,
        num=config["NUM_CHECKPOINTS"],
        endpoint=True,
        dtype=np.int32
        ))

    for key, val in config.items():
        print(key, val)
    print()
    print()

    # Use log scale for potentially better training results
    env = LogWrapper(env, replace_info=True)

    # Initialize TD3
    # We are assuming that the continuous action space is symmetric
    config["MAX_ACTION"] = env.action_spaces[env.agents[0]].high
    # rng_init, _rng = jax.random.split(rng_init)
    state_actor, state_actor_target, state_critic, state_critic_target = make_td3(rng_init, config, env)
    train_states = TrainStates(
        state_actor,
        state_actor_target,
        state_critic,
        state_critic_target
    )

    # Perform random exploration and do one single TD3 update
    # in the end, yes it will be quite a large update batch
    def _random_explore(rng, train_states, env_state, obsv):
        rng, _rng = jax.random.split(rng)
        explore_runner_state = _rng, env_state, obsv
        (
            explore_runner_state,
            (explore_trajectories, explore_reward)
        ) = jax.lax.scan(
            jax.jit(_make_random_env_step(config, env), device=jax.devices("cpu")[0]),
            explore_runner_state, None,
            length=config["EXPLORATION_ITERATIONS"]
        )
        _rng, env_state, obsv = explore_runner_state

        # Reshape to combine scanned dimensions and number of agents
        explore_trajectories = jax.tree.map(
            lambda x: jnp.reshape(x, (x.shape[0]*x.shape[1], ) + x.shape[2:] ),
            explore_trajectories
        )

        counter = 0
        rng, _rng = jax.random.split(rng)
        train_states, loss_info = make_td3_update(
            config
        )(_rng, train_states, explore_trajectories, counter)

        return train_states, loss_info, explore_reward, env_state, obsv

    def _train(rng: jax.dtypes.prng_key, train_states: TrainStates):
        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # Perform random exploration and do one single TD3 update
        rng, _rng = jax.random.split(rng)
        train_states, loss_info, explore_rewards, env_state, obsv = _random_explore(_rng, train_states, env_state, obsv)

        # From now perform online updates
        # The replay buffer as implemented in PyTorch is not feasible in Jax
        # Collect number of trajectories equal to batch size and then update using those
        counter = 0
        rng, _rng = jax.random.split(rng)
        runner_state = _rng, train_states, (env_state, obsv), counter
        runner_state, metrics = jax.lax.scan(
            make_iteration(config, env),
            runner_state, None,
            length=config["NUM_ITERATIONS"]
        )

        return {"runner_state": runner_state, "metrics": metrics}
    return train_states, _train

@hydra.main(version_base=None, config_path="config", config_name="test_td3")
def main(config):
    config = OmegaConf.to_container(config)

    rng = jax.random.PRNGKey(config["JAX_SEED"])
    rng, _rng = jax.random.split(rng)

    start_time = datetime.now()
    train_states, train_fn = make_train(config, _rng)
    train_jit = jax.jit(train_fn)
    stop_time = datetime.now()
    print(f"Jit Elapsed {stop_time-start_time}")
    file_write(
        os.path.join(ROOT_DIR, 'timing.csv'),
        f"Jit Time, {stop_time-start_time}\n",
        append=True
    )

    start_time = datetime.now()
    res = train_jit(rng, train_states)
    stop_time = datetime.now()
    print(f"Training Elapsed {stop_time-start_time}")
    file_write(
        os.path.join(ROOT_DIR, 'timing.csv'),
        f"Run Time, {stop_time-start_time}\n",
        append=True
    )

    metrics_y_actor_loss = np.array(res['metrics']['loss']['actor_loss'])
    # get how many non-nan values
    x_actor_loss = []
    y_actor_loss = []
    for i in range(metrics_y_actor_loss.shape[0]):
        if not np.isnan(metrics_y_actor_loss[i]):
            x_actor_loss.append(i)
            y_actor_loss.append(metrics_y_actor_loss[i])

    y_critic_loss = res['metrics']['loss']['critic_loss']

    loss_plot = LinePlot("Iteration", "Loss")
    loss_plot.add(np.arange(y_critic_loss.shape[0]), y_critic_loss, label="Critic Loss")
    loss_plot.add(x_actor_loss, y_actor_loss, label="Actor Loss")
    loss_plot.save(os.path.join(ROOT_DIR, "train_loss.png"))

    pickle_dump(
        os.path.join(DATA_DIR, 'mean-reward.pkl'),
        res['metrics']['reward']
    )


    reward_data = jax.tree.map(
        lambda x: x.reshape( (x.shape[0], -1) ),
        res['metrics']['reward']['__all__']
    )

    reward_plot = LinePlot("Iteration", "Mean Reward")
    reward_plot.add(np.arange(reward_data.shape[0]), jnp.mean(reward_data, axis=1))
    reward_plot.save(os.path.join(ROOT_DIR, 'mean-reward.png'))

    cumulative_reward_plot = LinePlot("Iteration", "Cumulative Mean Reward")
    cumulative_reward_plot.add(np.arange(reward_data.shape[0]), np.cumsum(jnp.mean(reward_data, axis=1)))
    cumulative_reward_plot.save(os.path.join(ROOT_DIR, 'cumulative-mean-reward.png'))

    print(os.listdir(ROOT_DIR))

if __name__ == "__main__":
    main()
