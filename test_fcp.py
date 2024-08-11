
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

from ficticious_coplay.util.util import nary_sequences


import jax

import pickle
from typing import Sequence

import distrax
import flax.linen as nn

import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.environments.overcooked.overcooked import DELIVERY_REWARD
import optax
import matplotlib.pyplot as plt

from ficticious_coplay.fcp import FCP, EnvSpec, SelfPlayAgent, TeamSpec, get_rollout


class SimpleNetwork(nn.Module):
    output_dim: Sequence[int] # action_space_dim
    activation = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            raise ValueError(f"Activation function for '{self.activation}' is not defined")
        x = nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)

        pi_logits = x
        pi_logits = nn.Dense(self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(pi_logits)
        pi = distrax.Categorical(logits=pi_logits)

        value = x
        value = nn.Dense(8, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = activation(value)
        value = nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value_squeezed = jnp.squeeze(value, axis=-1)

        return pi, value_squeezed
    


def make_ppo_agent(init_rng, config, env_spec: EnvSpec, team_spec: TeamSpec, env, checkpoint_prefix):
    _save_format_step = len(str(int((config["NUM_EPISODES"]*config["NUM_UPDATES"])-1)))
    network = SimpleNetwork(env.action_space().n)
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

    rng, _rng = jax.random.split(rng)
    start_time = datetime.now()
    stage_1_jit = FCP.make_stage_1( config, env_spec, teams, config["NUMPY_SEED"])
    stop_time = datetime.now()
    s1_time_jit = stop_time - start_time
    print(f"\nStage 1 Jit {s1_time_jit}")
    start_time = datetime.now()
    s1_episode_metrics, s1_last_episode_runner_state = stage_1_jit(_rng)
    stop_time = datetime.now()
    s1_time_run = stop_time - start_time
    print(f"\nStage 1 Elapsed {s1_time_run}")

    with open(os.path.join(ROOT_DIR, 'timing.csv'), 'a') as f:
        f.write(f"Stage 1 Jit Time, {s1_time_jit}\n")
        f.write(f"Stage 1 Run Time, {s1_time_run}\n")

    # Turn off Memory profiling in favour of performance
    # __profile_dir = os.path.join(ROOT_DIR, 'mem_profile')
    # os.makedirs(__profile_dir, exist_ok=True)
    # jax.profiler.save_device_memory_profile(os.path.join(__profile_dir, 'mem_stage_1.prof'))

    # Don't save this because the file is gigantic >10 GB on large runs
    # with open(os.path.join(DATA_DIR, 'stage-1_reward-metrics.pkl'), 'wb') as f:
    #     pickle.dump(s1_episode_metrics["reward"], f)

    flattened_cumulative_reward = jax.tree.map(
        (lambda x: jnp.cumsum(jnp.ravel(jnp.mean(x, axis=-1)))),
        s1_episode_metrics["reward"]
    )
    with open(os.path.join(DATA_DIR, 'stage-1_cumulative-reward.pkl'), 'wb') as f:
        pickle.dump(flattened_cumulative_reward, f)
    for team_ix, team_rewards in flattened_cumulative_reward.items():
        for p_ix, partner_rewards in team_rewards.items():
            plt.plot(range(partner_rewards.shape[0]), partner_rewards, label=f"{team_ix}-{p_ix}")
    plt.xlabel("Environment Step")
    plt.ylabel("Cumulative Mean Reward (mean over environment instances)")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-1_cumulative-mean-reward-per-partner.png"))
    plt.close()

    flattened_cumulative_dishes = jax.tree.map(
        (lambda x: jnp.cumsum(jnp.ravel(jnp.mean(jnp.isclose(x, DELIVERY_REWARD), axis=-1)))),
        s1_episode_metrics["reward"]
    )
    with open(os.path.join(DATA_DIR, 'stage-1_cumulative-delivered-dishes.pkl'), 'wb') as f:
        pickle.dump(flattened_cumulative_dishes, f)
    for team_ix, team_dishes in flattened_cumulative_dishes.items():
        for p_ix, partner_dishes in team_dishes.items():
            plt.plot(range(partner_dishes.shape[0]), partner_dishes, label=f"{team_ix}-{p_ix}")
    plt.xlabel("Environment Step")
    plt.ylabel("Cumulative Mean Dishes Delivered (mean over environment instances)")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-1_cumulative-mean-dishes-per-partner.png"))
    plt.close()

    total_update_steps = int(config["NUM_UPDATES"] * config["NUM_EPISODES"])
    flattened_loss = jax.tree.map(
        (lambda x: jnp.mean(x, axis=2).reshape((total_update_steps, ) + x.shape[2:])),
        s1_episode_metrics["update_metrics"]
        )
    with open(os.path.join(DATA_DIR, 'stage-1_loss-total.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[0], flattened_loss), f)
    with open(os.path.join(DATA_DIR, 'stage-1_loss-value.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[1][0], flattened_loss), f)
    with open(os.path.join(DATA_DIR, 'stage-1_loss-actor.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[1][1], flattened_loss), f)
    with open(os.path.join(DATA_DIR, 'stage-1_loss-entropy.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[1][2], flattened_loss), f)
    for team_ix, team_metrics in flattened_loss.items():
        for p_ix, partner_metrics in team_metrics.items():
            plt.plot(range(total_update_steps), partner_metrics[0], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Total Loss")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-1_loss-total.png"))
    plt.close()
    for team_ix, team_metrics in flattened_loss.items():
        for p_ix, partner_metrics in team_metrics.items():
            plt.plot(range(total_update_steps), partner_metrics[1][0], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Value Loss")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-1_loss-value.png"))
    plt.close()
    for team_ix, team_metrics in flattened_loss.items():
        for p_ix, partner_metrics in team_metrics.items():
            plt.plot(range(total_update_steps), partner_metrics[1][1], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Actor Loss")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-1_loss-actor.png"))
    plt.close()
    for team_ix, team_metrics in flattened_loss.items():
        for p_ix, partner_metrics in team_metrics.items():
            plt.plot(range(total_update_steps), partner_metrics[1][2], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Entropy")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-1_loss-entropy.png"))
    plt.close()




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
    rng, _rng = jax.random.split(rng)
    start_time = datetime.now()
    stage_2_jit = FCP.make_stage_2(
        config, env_spec, teams,
        team_fcp_agents,
        load_steps,
        config["NUMPY_SEED"]
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
        f.write(f"Stage 1 Jit Time, {s2_time_jit}\n")
        f.write(f"Stage 1 Run Time, {s2_time_run}\n")

    # Turn off Memory profiling in favour of performance
    # __profile_dir = os.path.join(ROOT_DIR, 'mem_profile')
    # os.makedirs(__profile_dir, exist_ok=True)
    # jax.profiler.save_device_memory_profile(os.path.join(__profile_dir, 'mem_stage_2.prof'))

    flattened_cumulative_reward = jax.tree.map(
        (lambda x: jnp.cumsum(jnp.ravel(jnp.mean(x, axis=-1)))),
        s2_episode_metrics["reward"]
    )
    for team_ix, team_rewards in flattened_cumulative_reward.items():
        for p_ix, partner_rewards in team_rewards.items():
            alpha=0.2
            label = ""
            if team_fcp_agents[team_ix]:
                if p_ix == 0:
                    label = f"{team_ix}-fcp"
                    alpha=1
            plt.plot(range(partner_rewards.shape[0]), partner_rewards, label=label, alpha=alpha)
    plt.xlabel("Environment Step")
    plt.ylabel("Cumulative Mean Reward (mean over environment instances)")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-2_cumulative_mean_reward_per_partner.png"))
    plt.close()

    flattened_cumulative_dishes = jax.tree.map(
        (lambda x: jnp.cumsum(jnp.ravel(jnp.mean(jnp.isclose(x, DELIVERY_REWARD), axis=-1)))),
        s2_episode_metrics["reward"]
    )
    for team_ix, team_dishes in flattened_cumulative_dishes.items():
        for p_ix, partner_dishes in team_dishes.items():
            alpha=0.2
            label = ""
            if team_fcp_agents[team_ix]:
                if p_ix == 0:
                    label = f"{team_ix}-fcp"
                    alpha=1
            plt.plot(range(partner_dishes.shape[0]), partner_dishes, label=label, alpha=alpha)
    plt.xlabel("Environment Step")
    plt.ylabel("Cumulative Mean Dishes Delivered (mean over environment instances)")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-2_cumulative_mean_dishes_per_partner.png"))
    plt.close()


    total_update_steps = int(config["NUM_UPDATES"] * config["NUM_EPISODES"])
    flattened_loss = jax.tree.map(
        (lambda x: jnp.mean(x, axis=2).reshape((total_update_steps, ) + x.shape[2:])),
        s2_episode_metrics["update_metrics"]
        )
    with open(os.path.join(DATA_DIR, 'stage-2_loss-total.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[0], flattened_loss), f)
    with open(os.path.join(DATA_DIR, 'stage-2_loss-value.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[1][0], flattened_loss), f)
    with open(os.path.join(DATA_DIR, 'stage-2_loss-actor.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[1][1], flattened_loss), f)
    with open(os.path.join(DATA_DIR, 'stage-2_loss-entropy.pkl'), 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x[1][2], flattened_loss), f)
    for team_ix, team_metrics in flattened_loss.items():
        if team_fcp_agents[team_ix]:
            p_ix, partner_metrics = 0, team_metrics[0]
            plt.plot(range(total_update_steps), partner_metrics[0], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Total Loss")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-2_loss-total.png"))
    plt.close()
    for team_ix, team_metrics in flattened_loss.items():
        if team_fcp_agents[team_ix]:
            p_ix, partner_metrics = 0, team_metrics[0]
            plt.plot(range(total_update_steps), partner_metrics[1][0], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Value Loss")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-2_loss-value.png"))
    plt.close()
    for team_ix, team_metrics in flattened_loss.items():
        if team_fcp_agents[team_ix]:
            p_ix, partner_metrics = 0, team_metrics[0]
            plt.plot(range(total_update_steps), partner_metrics[1][1], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Actor Loss")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-2_loss-actor.png"))
    plt.close()
    for team_ix, team_metrics in flattened_loss.items():
        if team_fcp_agents[team_ix]:
            p_ix, partner_metrics = 0, team_metrics[0]
            plt.plot(range(total_update_steps), partner_metrics[1][2], label=f"{team_ix}-{p_ix}")
    plt.xlabel("Entropy")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, "stage-2_loss-entropy.png"))
    plt.close()




def __rollout_permutation(
        config, rng, rollout_env_spec, team_agents, rollout_teams,
        rollout_permutation, max_rollout_steps,
        rollout_reward_matrices, rollout_dishes_matrices
    ):
    seq_envs_states, seq_envs_rewards = get_rollout(config, rng, rollout_env_spec, team_agents, rollout_permutation, max_steps=max_rollout_steps)
    cumulative_reward_per_agent = jax.tree.map(lambda x: jnp.sum(jnp.mean(x, axis=1)), seq_envs_rewards)
    delivered_dishes_per_agent = jax.tree.map(lambda x: jnp.sum(jnp.mean(jnp.isclose(x, DELIVERY_REWARD), axis=1)), seq_envs_rewards)
    for team_ix, team_permutation in enumerate(rollout_permutation):
        # TODO we're being lazy here because we know the current overcooked environment has only 2 agents
        a0, a1 = team_permutation[0], team_permutation[1]
        ix_a0 = rollout_teams[team_ix].index(a0)
        ix_a1 = rollout_teams[team_ix].index(a1)
        rollout_reward_matrices[team_ix][ix_a0, ix_a1] = cumulative_reward_per_agent["agent_0"]+cumulative_reward_per_agent["agent_1"]
        rollout_dishes_matrices[team_ix][ix_a0, ix_a1] = delivered_dishes_per_agent["agent_0"]+delivered_dishes_per_agent["agent_1"]

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
    rollout_reward_matrices = []
    rollout_dishes_matrices = []
    for rollout_team in rollout_teams:
        rollout_reward_matrices.append(np.full((len(rollout_team), len(rollout_team)), -1.0))
        rollout_dishes_matrices.append(np.full((len(rollout_team), len(rollout_team)), -1.0))

    for rollout_permutation in rollout_permutations:
        rng, _rng = jax.random.split(rng)
        __rollout_permutation(
            config, _rng, rollout_env_spec, team_agents, rollout_teams,
            rollout_permutation, max_rollout_steps,
            rollout_reward_matrices, rollout_dishes_matrices
            )

    for team_ix in range(len(rollout_reward_matrices)):
        labels = [ prefix.replace(f"{team_ix}-", "")+f"{ckpt}" for (prefix, ckpt), _ in rollout_teams[team_ix] ]
        fig, ax = plt.subplots(figsize=[30, 30])
        ax.matshow(rollout_reward_matrices[team_ix], cmap=plt.cm.Blues)
        for i in range(rollout_reward_matrices[team_ix].shape[0]):
            for j in range(rollout_reward_matrices[team_ix].shape[1]):
                c = np.round(rollout_reward_matrices[team_ix][i,j], 2)
                ax.text(i, j, str(c), va='center', ha='center')
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        fig.savefig(os.path.join(ROOT_DIR, f"cumulative-reward_team-{team_ix}.png"))
        plt.close(fig)

        labels = [ prefix.replace(f"{team_ix}-", "")+f"{ckpt}" for (prefix, ckpt), _ in rollout_teams[team_ix] ]
        fig, ax = plt.subplots(figsize=[30, 30])
        ax.matshow(rollout_dishes_matrices[team_ix], cmap=plt.cm.Blues)
        for i in range(rollout_dishes_matrices[team_ix].shape[0]):
            for j in range(rollout_dishes_matrices[team_ix].shape[1]):
                c = np.round(rollout_dishes_matrices[team_ix][i,j], 2)
                ax.text(i, j, str(c), va='center', ha='center')
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        fig.savefig(os.path.join(ROOT_DIR, f"cumulative-delivered-dishes_team-{team_ix}.png"))
        plt.close(fig)





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
