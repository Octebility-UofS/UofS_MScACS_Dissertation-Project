
from itertools import combinations, permutations
import os
# Recommended XLA flags for improving gpu performance
# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#xla-performance-flags
#os.environ['XLA_FLAGS'] = (
#    '--xla_gpu_enable_triton_softmax_fusion=true '
#    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
#)
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']= 'false'

import __main__
import sys
from datetime import datetime
__script_name = ".".join(os.path.split(__main__.__file__)[1].split(".")[:-1])
__time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT_DIR = os.path.join('.', 'out', f"0_{__time}_{__script_name}")
if __name__ == "__main__":
    if any([ arg.startswith("hydra.run.dir=") for arg in sys.argv ]):
        arg_ix = [ _ix for _ix, arg in enumerate(sys.argv) if arg.startswith("hydra.run.dir=") ][0]
        _arg_dirpath = sys.argv.pop(arg_ix).replace("hydra.run.dir=", "")
        ROOT_DIR = os.path.join('.', 'out', _arg_dirpath + f"_{__script_name}")
        sys.argv.append(f'hydra.run.dir={ROOT_DIR}/hydra')
    elif any([ arg.startswith("JOB_ID=") for arg in sys.argv ]):
        arg_ix = [ _ix for _ix, arg in enumerate(sys.argv) if arg.startswith("JOB_ID=") ][0]
        _arg_job_id = sys.argv.pop(arg_ix).replace("JOB_ID=", "")
        ROOT_DIR = os.path.join('.', 'out', f"{_arg_job_id}_{__time}_{__script_name}")
        sys.argv.append(f'hydra.run.dir={ROOT_DIR}/hydra')
    else:
        sys.argv.append(f'hydra.run.dir={ROOT_DIR}/hydra')
os.makedirs(ROOT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

import hydra
from omegaconf import OmegaConf

from ficticious_coplay.rollout import make_rollout
from ficticious_coplay.common import SelfPlayAgent
from util.util import HeatMatrix, LinePlot, file_write, nary_sequences, pickle_dump


import jax

import pickle
from typing import Sequence

import distrax
import flax.linen as nn

import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.environments.overcooked.overcooked import DELIVERY_REWARD
import optax

from ficticious_coplay.fcp import FCP, EnvSpec, TeamSpec

class OvercookedMLPActorCritic(nn.Module):
    output_dim: Sequence[int] # action_space_dim

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        value = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        value = nn.relu(value)
        value = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = nn.relu(value)
        value = nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)

        return pi, jnp.squeeze(value, axis=-1)


# class SimpleNetwork(nn.Module):
#     output_dim: Sequence[int] # action_space_dim
#     activation = "relu"

#     @nn.compact
#     def __call__(self, x):
#         if self.activation == "relu":
#             activation = nn.relu
#         else:
#             raise ValueError(f"Activation function for '{self.activation}' is not defined")
#         x = nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
#         x = activation(x)

#         pi_logits = x
#         pi_logits = nn.Dense(self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(pi_logits)
#         pi = distrax.Categorical(logits=pi_logits)

#         value = x
#         value = nn.Dense(8, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
#         value = activation(value)
#         value = nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
#         value_squeezed = jnp.squeeze(value, axis=-1)

#         return pi, value_squeezed
    


def make_ppo_agent(init_rng, config, env_spec: EnvSpec, team_spec: TeamSpec, env, checkpoint_prefix):
    _save_format_step = len(str(int((config["NUM_EPISODES"]*config["NUM_UPDATES"])-1)))
    network = OvercookedMLPActorCritic(env.action_space().n)
    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()
    init_rng, _rng = jax.random.split(init_rng)
    network_params = network.init(_rng, init_x)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["ENV_STEPS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac
    tx = None
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5)
            )

    init_agent_state = {}
    init_agent_state["train_state"] = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    @jax.jit
    def get_action(rng, obsv, flattened_obsv, agent_state):
        pi, value = network.apply(agent_state["train_state"].params, flattened_obsv)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        return agent_state, action, (pi, value, log_prob)
    

    # This actually seems to be IPPO loss
    # TODO
    @jax.jit
    def _loss_fn(net_train_state_params, traj_batch, gae, targets):
        # net_train_state_params = agent_state["train_state"].params
        # gae = Advantages
        # RERUN NETWORK
        # Since we are working in flattned minibatches batches, we only need to keep the first dimension
        flattened_obs = traj_batch.obs.reshape((traj_batch.obs.shape[0], -1))
        pi, value = network.apply(net_train_state_params, flattened_obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        # traj_batch.processed_observation[1] == traj_batch.value
        value_pred_clipped = traj_batch.processed_observation[1] + (
            value - traj_batch.processed_observation[1]
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # CALCULATE ACTOR LOSS
        # traj_batch.processed_observation[2] == traj_batch.log_prob
        ratio = jnp.exp(log_prob - traj_batch.processed_observation[2])
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)
    
    @jax.jit
    def _update_minibatch(agent_state, batch_data):
        trajectories, advantages, targets = batch_data

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, gradients = grad_fn(agent_state["train_state"].params, trajectories, advantages, targets)

        updated_train_state = agent_state["train_state"].apply_gradients(grads=gradients)
        update_agent_state = {
            "train_state": updated_train_state
        }
        return update_agent_state, total_loss


    @jax.jit
    def update(rng, trajectories, agent_state):
        done, actions, aux_data, reward, observations, info = trajectories

        # CALCULATE ADVANTAGE
        # This seems to be the same for PPO and IPPO
        flattened_last_obsv = observations[-1].reshape((observations[-1].shape[0], -1))
        last_val = network.apply(agent_state["train_state"].params, flattened_last_obsv)[1]

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.processed_observation[1],
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.processed_observation[1] # advantages + values

        advantages, targets = _calculate_gae(trajectories, last_val)
        assert ( (config["ENV_STEPS"] % config["NUM_MINIBATCHES"]) == 0 ), "Number of Minibatches must be divisible by number of environment steps"
        batch_size = advantages.shape[0] * advantages.shape[1]
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, batch_size)
        batch = (trajectories, advantages, targets)
        batch_reshaped = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        )
        batch_shuffled = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch_reshaped
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
            ),
            batch_shuffled,
        )
        
        updated_agent_state, total_loss = jax.lax.scan(
            _update_minibatch, agent_state, minibatches
        )
        return updated_agent_state, total_loss
    
    def save(agent_state, step):
        if not (step in config["_CHECKPOINT_STEPS"]):
            return agent_state
        # I guess it's best to pickle and unpickle since the other things don't really seem to work
        # Only save network parameters to save disk space
        target = agent_state["train_state"].params
        str_step = str(int(step)).zfill(_save_format_step)
        with open(os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}{str_step}.param.ckpt"), 'wb') as f:
            pickle.dump(target, f)
        return agent_state

    def load(agent_state, step):
        def _frozen_update(rng, trajectories, agent_state):
            return agent_state, (jnp.zeros(1), (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)))
        def _frozen_save(agent_state, step):
            return agent_state
        
        restored_params = None
        str_step = str(int(step)).zfill(_save_format_step)
        with open(os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}{str_step}.param.ckpt"), 'rb') as f:
            restored_params = pickle.load(f)

        new_agent_state = {}
        new_agent_state["train_state"] = TrainState.create(
            apply_fn=network.apply,
            params=restored_params,
            tx=tx
        )
        return new_agent_state, _frozen_update, _frozen_save
    
    return SelfPlayAgent(
        get_action,
        update,
        save,
        load,
    ), init_agent_state


























def _process_stage_1(config, rng):
    # This is part of the config
    # All environments specified here must have the same action space and observation space dimensions
    env_spec = EnvSpec(
        config["ENV"]["ID"],
        config["ENV"]["COUNT"],
        {"layout": overcooked_layouts[config["ENV"]["KWARGS"]["layout"]]}
    )
    teams = []
    for _team_ix, team_config in config["TEAMS"].items():
        teams.append(TeamSpec(
            globals().get(team_config["CLS_AGENT"]),
            team_config["AGENT_COUNT"],
            team_config["AGENT_IDS"]
        ))

    rng, init_rng = jax.random.split(rng)
    rng, _rng = jax.random.split(rng)
    start_time = datetime.now()
    stage_1_jit = FCP.make_stage_1(config, init_rng, env_spec, teams)
    stop_time = datetime.now()
    s1_time_jit = stop_time - start_time
    print(f"\nStage 1 Jit {s1_time_jit}")
    start_time = datetime.now()
    s1_episode_metrics, s1_last_episode_runner_state = stage_1_jit(_rng)
    stop_time = datetime.now()
    s1_time_run = stop_time - start_time
    print(f"\nStage 1 Elapsed {s1_time_run}")

    file_write(
        os.path.join(ROOT_DIR, 'timing.csv'),
        f"Stage 1 Jit Time, {s1_time_jit}\n",
        append=True
        )
    file_write(
        os.path.join(ROOT_DIR, 'timing.csv'),
        f"Stage 1 Run Time, {s1_time_run}\n",
        append=True
        )

    # Turn off Memory profiling in favour of performance
    # __profile_dir = os.path.join(ROOT_DIR, 'mem_profile')
    # os.makedirs(__profile_dir, exist_ok=True)
    # jax.profiler.save_device_memory_profile(os.path.join(__profile_dir, 'mem_stage_1.prof'))


    # -> Reshape to put combine episodes and update steps
    # -> take mean across environments
    # -> combine into cumulative sum
    cumulative_reward_per_update = jax.tree.map(
        lambda x: jnp.cumsum(jnp.mean(x.reshape((int(config["NUM_EPISODES"]*config["NUM_UPDATES"]), ) + x.shape[2:]), axis=-1)),
        s1_episode_metrics["reward"]
    )
    cumulative_delivered_dishes_per_update = jax.tree.map(
        lambda x: jnp.cumsum(jnp.mean(x.reshape((int(config["NUM_EPISODES"]*config["NUM_UPDATES"]), ) + x.shape[2:]) == DELIVERY_REWARD, axis=-1)),
        s1_episode_metrics["reward"]
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-1_cumulative-reward.pkl'),
        cumulative_reward_per_update
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-1_cumulative-delivered-dishes.pkl'),
        cumulative_delivered_dishes_per_update
    )
    reward_plot = LinePlot("Environment Step", "Cumulative Mean Reward")
    dishes_plot = LinePlot("Environment Step", "Cumulative Mean Dishes Delivered")
    for team_ix, team_rewards in cumulative_reward_per_update.items():
        for p_ix, partner_rewards in team_rewards.items():
            reward_plot.add(range(partner_rewards.shape[0]), partner_rewards, label=f"{team_ix}-{p_ix}")
    for team_ix, team_dishes in cumulative_delivered_dishes_per_update.items():
        for p_ix, partner_dishes in team_dishes.items():
            dishes_plot.add(range(partner_dishes.shape[0]), partner_dishes, label=f"{team_ix}-{p_ix}")
    reward_plot.save(os.path.join(ROOT_DIR, "stage-1_cumulative-mean-reward-per-partner.png"))
    dishes_plot.save(os.path.join(ROOT_DIR, "stage-1_cumulative-mean-dishes-per-partner.png"))
    
    

    total_update_steps = int(config["NUM_UPDATES"] * config["NUM_EPISODES"])
    flattened_loss = jax.tree.map(
        (lambda x: jnp.mean(x, axis=2).reshape((total_update_steps, ) + x.shape[2:])),
        s1_episode_metrics["update_metrics"]
        )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-1_loss-total.pkl'),
        jax.tree.map(lambda x: x[0], flattened_loss)
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-1_loss-value.pkl'),
        jax.tree.map(lambda x: x[1][0], flattened_loss)
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-1_loss-actor.pkl'),
        jax.tree.map(lambda x: x[1][1], flattened_loss)
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-1_loss-entropy.pkl'),
        jax.tree.map(lambda x: x[1][2], flattened_loss)
    )

    total_loss_plot = LinePlot("Update Step", "Total Loss")
    value_loss_plot = LinePlot("Update Step", "Value Loss")
    actor_loss_plot = LinePlot("Update Step", "Actor Loss")
    entropy_loss_plot = LinePlot("Update Step", "Entropy")
    for team_ix, team_metrics in flattened_loss.items():
        for p_ix, partner_metrics in team_metrics.items():
            total_loss_plot.add(range(total_update_steps), partner_metrics[0], label=f"{team_ix}-{p_ix}")
            value_loss_plot.add(range(total_update_steps), partner_metrics[1][0], label=f"{team_ix}-{p_ix}")
            actor_loss_plot.add(range(total_update_steps), partner_metrics[1][1], label=f"{team_ix}-{p_ix}")
            entropy_loss_plot.add(range(total_update_steps), partner_metrics[1][2], label=f"{team_ix}-{p_ix}")
    total_loss_plot.save(os.path.join(ROOT_DIR, "stage-1_loss-total.png"))
    value_loss_plot.save(os.path.join(ROOT_DIR, "stage-1_loss-value.png"))
    actor_loss_plot.save(os.path.join(ROOT_DIR, "stage-1_loss-actor.png"))
    entropy_loss_plot.save(os.path.join(ROOT_DIR, "stage-1_loss-entropy.png"))
























def _process_stage_2(config, rng):
    env_spec = EnvSpec(
        config["ENV"]["ID"],
        config["ENV"]["COUNT"],
        {"layout": overcooked_layouts[config["ENV"]["KWARGS"]["layout"]]}
    )
    teams = []
    for _team_ix, team_config in config["TEAMS"].items():
        teams.append(TeamSpec(
            globals().get(team_config["CLS_AGENT"]),
            team_config["AGENT_COUNT"],
            team_config["AGENT_IDS"]
        ))
    saved_steps = config["_CHECKPOINT_STEPS"]
    load_steps = [saved_steps[0], saved_steps[len(saved_steps)//2], saved_steps[-1]]
    team_fcp_agents = [ globals().get(fcp_agent_cls_str) for fcp_agent_cls_str in config["FCP_AGENTS"] ]
    rng, init_rng = jax.random.split(rng)
    rng, _rng = jax.random.split(rng)
    start_time = datetime.now()
    stage_2_jit = FCP.make_stage_2(
        config, init_rng, env_spec, teams,
        team_fcp_agents,
        load_steps
        )
    stop_time = datetime.now()
    s2_time_jit = stop_time - start_time
    print(f"\nStage 2 Jit {s2_time_jit}")
    start_time = datetime.now()
    s2_episode_metrics, s2_last_episode_runner_state = stage_2_jit(_rng)
    stop_time = datetime.now()
    s2_time_run = stop_time - start_time
    print(f"\nStage 2 Elapsed {s2_time_run}")

    with open(os.path.join(ROOT_DIR, 'timing.csv'), 'a') as f:
        f.write(f"Stage 2 Jit Time, {s2_time_jit}\n")
        f.write(f"Stage 2 Run Time, {s2_time_run}\n")

    # Turn off Memory profiling in favour of performance
    # __profile_dir = os.path.join(ROOT_DIR, 'mem_profile')
    # os.makedirs(__profile_dir, exist_ok=True)
    # jax.profiler.save_device_memory_profile(os.path.join(__profile_dir, 'mem_stage_2.prof'))

    # -> Reshape to put combine episodes and update steps
    # -> take mean across environments
    # -> combine into cumulative sum
    cumulative_reward_per_update = jax.tree.map(
        lambda x: jnp.cumsum(jnp.mean(x.reshape((int(config["NUM_EPISODES"]*config["NUM_UPDATES"]), ) + x.shape[2:]), axis=-1)),
        s2_episode_metrics["reward"]
    )
    cumulative_delivered_dishes_per_update = jax.tree.map(
        lambda x: jnp.cumsum(jnp.mean(x.reshape((int(config["NUM_EPISODES"]*config["NUM_UPDATES"]), ) + x.shape[2:]) == DELIVERY_REWARD, axis=-1)),
        s2_episode_metrics["reward"]
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-2_cumulative-reward.pkl'),
        cumulative_reward_per_update
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-2_cumulative-delivered-dishes.pkl'),
        cumulative_delivered_dishes_per_update
    )
    reward_plot = LinePlot("Environment Step", "Cumulative Mean Reward")
    dishes_plot = LinePlot("Environment Step", "Cumulative Mean Dishes Delivered")
    for team_ix, team_rewards in cumulative_reward_per_update.items():
        for p_ix, partner_rewards in team_rewards.items():
            if p_ix == 0 and team_fcp_agents[team_ix]:
                reward_plot.add(range(partner_rewards.shape[0]), partner_rewards, label=f"{team_ix}-fcp", alpha=1.0)
            else:
                reward_plot.add(range(partner_rewards.shape[0]), partner_rewards, alpha=0.2)
    for team_ix, team_dishes in cumulative_delivered_dishes_per_update.items():
        for p_ix, partner_dishes in team_dishes.items():
            if p_ix == 0 and team_fcp_agents[team_ix]:
                dishes_plot.add(range(partner_dishes.shape[0]), partner_dishes, label=f"{team_ix}-fcp", alpha=1.0)
            else:
                dishes_plot.add(range(partner_dishes.shape[0]), partner_dishes, alpha=0.2)
    reward_plot.save(os.path.join(ROOT_DIR, "stage-2_cumulative-mean-reward-per-partner.png"))
    dishes_plot.save(os.path.join(ROOT_DIR, "stage-2_cumulative-mean-dishes-per-partner.png"))


    total_update_steps = int(config["NUM_UPDATES"] * config["NUM_EPISODES"])
    flattened_loss = jax.tree.map(
        (lambda x: jnp.mean(x, axis=2).reshape((total_update_steps, ) + x.shape[2:])),
        s2_episode_metrics["update_metrics"]
        )

    pickle_dump(
        os.path.join(DATA_DIR, 'stage-2_loss-total.pkl'),
        jax.tree.map(lambda x: x[0], flattened_loss)
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-2_loss-value.pkl'),
        jax.tree.map(lambda x: x[1][0], flattened_loss)
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-2_loss-actor.pkl'),
        jax.tree.map(lambda x: x[1][1], flattened_loss)
    )
    pickle_dump(
        os.path.join(DATA_DIR, 'stage-2_loss-entropy.pkl'),
        jax.tree.map(lambda x: x[1][2], flattened_loss)
    )

    total_loss_plot = LinePlot("Update Step", "Total Loss")
    value_loss_plot = LinePlot("Update Step", "Value Loss")
    actor_loss_plot = LinePlot("Update Step", "Actor Loss")
    entropy_loss_plot = LinePlot("Update Step", "Entropy")
    for team_ix, team_metrics in flattened_loss.items():
        if team_fcp_agents[team_ix]:
            p_ix, partner_metrics = 0, team_metrics[0]
            total_loss_plot.add(range(total_update_steps), partner_metrics[0], label=f"{team_ix}-{p_ix}")
            value_loss_plot.add(range(total_update_steps), partner_metrics[1][0], label=f"{team_ix}-{p_ix}")
            actor_loss_plot.add(range(total_update_steps), partner_metrics[1][1], label=f"{team_ix}-{p_ix}")
            entropy_loss_plot.add(range(total_update_steps), partner_metrics[1][2], label=f"{team_ix}-{p_ix}")
    total_loss_plot.save(os.path.join(ROOT_DIR, "stage-2_loss-total.png"))
    value_loss_plot.save(os.path.join(ROOT_DIR, "stage-2_loss-value.png"))
    actor_loss_plot.save(os.path.join(ROOT_DIR, "stage-2_loss-actor.png"))
    entropy_loss_plot.save(os.path.join(ROOT_DIR, "stage-2_loss-entropy.png"))




















def _process_rollout(config, rng):
    max_rollout_steps = config["ROLLOUT_STEPS"]
    num_rollout_envs = config["ROLLOUT_NUM_ENVS"]
    rollout_env_spec = EnvSpec(
        config["ENV"]["ID"],
        num_rollout_envs,
        {"layout": overcooked_layouts[config["ENV"]["KWARGS"]["layout"]]}
    )
    teams = []
    for _team_ix, team_config in config["TEAMS"].items():
        teams.append(TeamSpec(
            globals().get(team_config["CLS_AGENT"]),
            team_config["AGENT_COUNT"],
            team_config["AGENT_IDS"]
        ))
    saved_steps = config["_CHECKPOINT_STEPS"]
    load_steps = [saved_steps[0], saved_steps[len(saved_steps)//2], saved_steps[-1]]
    team_fcp_agents = [ globals().get(fcp_agent_cls_str) for fcp_agent_cls_str in config["FCP_AGENTS"] ]

    rollout_teams = []
    team_agents = []
    for team_ix, team_spec in enumerate(teams):
        rollout_teams.append([])
        team_agents.append(team_spec.agent_uids)
        if team_fcp_agents[team_ix]:
            rollout_teams[team_ix].append( ((f"fcp-{team_ix}_", load_steps[-1]), team_fcp_agents[team_ix]) )
        for p_ix in range(team_spec.agent_count):
            for load_step in load_steps:
                rollout_teams[team_ix].append( ((f"{team_ix}-{p_ix}_", load_step), team_spec.agent_class) )
    team_permutations = []
    for rollout_team in rollout_teams:
        # TODO Once again we're being lazy by assuming that we only have 2 agents
        # Add diagonals (e.g. fcp vs fcp)
        team_permutations.append(
            list(permutations(rollout_team, len(team_spec.agent_uids)))
            + [ (t, t) for t in rollout_team ]
            )

    rollout_permutations = nary_sequences(*team_permutations)
    rng, init_rng = jax.random.split(rng)
    rng, _rng = jax.random.split(rng)
    scanned_states, scanned_rewards = jax.jit(make_rollout(
        config, init_rng, rollout_env_spec, team_agents, rollout_permutations, max_steps=max_rollout_steps
    ))(_rng)

    cumulative_reward = jax.tree.map(
        lambda x: jnp.sum(jnp.mean(x, axis=2), axis=0),
        scanned_rewards
    )
    cumulative_delivered_dishes = jax.tree.map(
        lambda x: jnp.sum(jnp.mean(x == DELIVERY_REWARD, axis=2), axis=0),
        scanned_rewards
    )

    for team_ix, team_partners in enumerate(rollout_teams):
        labels = [ str(t[0][0])+str(t[0][1]) for t in team_partners ]
        map_partner_to_index = { str(t[0][0])+str(t[0][1]): i for i, t in enumerate(team_partners) }
        for agent_uid_ix_0, agent_uid_ix_1 in combinations(range(len(teams[team_ix].agent_uids)), 2):
            agent_uid_0 = teams[team_ix].agent_uids[agent_uid_ix_0]
            agent_uid_1 = teams[team_ix].agent_uids[agent_uid_ix_1]
            save_name = f"_team-{team_ix}.{agent_uid_0}.{agent_uid_1}.png"
            matrix_reward = np.full((len(labels), len(labels)), -1.0)
            matrix_delivered_dishes = np.full((len(labels), len(labels)), -1.0)
            for rollout_ix, rollout_permutation in enumerate(rollout_permutations):
                p0_tup = rollout_permutation[team_ix][agent_uid_ix_0][0]
                p1_tup = rollout_permutation[team_ix][agent_uid_ix_1][0]
                p0_map_ix = map_partner_to_index[str(p0_tup[0])+str(p0_tup[1])]
                p1_map_ix = map_partner_to_index[str(p1_tup[0])+str(p1_tup[1])]
                combined_reward = cumulative_reward[agent_uid_0][rollout_ix] + cumulative_reward[agent_uid_1][rollout_ix]
                combined_dishes = cumulative_delivered_dishes[agent_uid_0][rollout_ix] + cumulative_delivered_dishes[agent_uid_1][rollout_ix]
                matrix_reward[p0_map_ix, p1_map_ix] = combined_reward
                matrix_delivered_dishes[p0_map_ix, p1_map_ix] = combined_dishes
            HeatMatrix(
                matrix_reward, labels, labels, figsize=[32, 32]
            ).save(os.path.join(ROOT_DIR, "cumulative-reward") + save_name)
            HeatMatrix(
                matrix_delivered_dishes, labels, labels, figsize=[32, 32]
            ).save(os.path.join(ROOT_DIR, "cumulative-delivered-dishes") + save_name)





@hydra.main(version_base=None, config_path="config", config_name="test_fcp")
def main(config):
    config = OmegaConf.to_container(config)

    config["_CHECKPOINT_STEPS"] = list(np.linspace(
        0,
        (config["NUM_UPDATES"] * config["NUM_EPISODES"]) - 1,
        num=config["NUM_CHECKPOINTS"],
        endpoint=True,
        dtype=np.int32
        ))    

    rng = jax.random.PRNGKey(config["JAX_SEED"])


    rng, _rng = jax.random.split(rng)
    _process_stage_1(config, _rng)

    rng, _rng = jax.random.split(rng)
    _process_stage_2(config, _rng)

    rng, _rng = jax.random.split(rng)
    _process_rollout(config, _rng)

    return None

    


if __name__ == "__main__":
    main()
