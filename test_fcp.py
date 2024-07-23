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

import sys
JOB_ID = "0"
if len(sys.argv) > 1:
    JOB_ID = sys.argv[1]
ROOT_DIR = os.path.join('.', 'out', JOB_ID)
os.makedirs(ROOT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import jax
# Cheap way of detecting whether this was called from my slurm job
#if len(sys.argv) > 1:
#    # Recommended for Slurm environment
#    jax.distributed.initialize()

from datetime import datetime

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
import optax
import matplotlib.pyplot as plt

from ficticious_coplay.fcp import FCP, EnvSpec, SelfPlayAgent, TeamSpec, _make_stage_2, get_rollout


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



def main():
    config = {
        "NUM_CHECKPOINTS": 100,
        "ENV_STEPS": 1e5,
        "NUM_UPDATES": 100,
        "NUM_MINIBATCHES": 10,
        "NUM_EPISODES": 1,
        "ANNEAL_LR": True,
        "MAX_GRAD_NORM": 0.5,
        "LR": 2.5e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        # "NUM_ENVS": 3, # 200
        # "NUM_AGENTS": 3, # 32
        # "NUM_STEPS": 25
    }
    config["_CHECKPOINT_STEPS"] = list(np.linspace(
        0,
        (config["NUM_UPDATES"] * config["NUM_EPISODES"]) - 1,
        num=config["NUM_CHECKPOINTS"],
        endpoint=True,
        dtype=np.int32
        ))

    rng = jax.random.PRNGKey(0)
    numpy_seed = 0

    # This is part of the config
    # All environments specified here must have the same action space and observation space dimensions
    env_spec = EnvSpec("overcooked", 200, {"layout" : overcooked_layouts["cramped_room"]})
    teams = [ TeamSpec(make_ppo_agent, 8, ['agent_0', 'agent_1']), ]

    rng, _rng = jax.random.split(rng)
    stage_1_jit = FCP.make_stage_1( config, env_spec, teams, numpy_seed)
    # stage_1_jit = jax.jit(FCP.make_stage_1(config, env_mapping, numpy_seed))
    start_time = datetime.now()
    s1_episode_metrics, s1_last_episode_runner_state = stage_1_jit(_rng)
    stop_time = datetime.now()
    print(f"Stage 1 Elapsed {stop_time-start_time}")

    __profile_dir = os.path.join(ROOT_DIR, 'mem_profile')
    os.makedirs(__profile_dir, exist_ok=True)
    jax.profiler.save_device_memory_profile(os.path.join(__profile_dir, 'mem_stage_1.prof'))


    total_update_steps = int(config["NUM_UPDATES"] * config["NUM_EPISODES"])
    for team_ix, team_metrics in s1_episode_metrics.items():
        for p_ix, partner_metrics in team_metrics.items():
            plt.plot(range(total_update_steps), partner_metrics[0], label=f"{team_ix}-{p_ix}")
    plt.legend()
    plt.savefig(f"./out/{JOB_ID}/stage-1_loss_per_partner.png")
    plt.close()




    saved_steps = config["_CHECKPOINT_STEPS"]
    load_steps = [saved_steps[0], saved_steps[len(saved_steps)//2], saved_steps[-1]]
    team_fcp_agents = [make_ppo_agent, ]
    rng, _rng = jax.random.split(rng)
    stage_2_jit = _make_stage_2(
        config, env_spec, teams,
        team_fcp_agents,
        load_steps,
        numpy_seed
        )
    start_time = datetime.now()
    s2_episode_metrics, s2_last_episode_runner_state = stage_2_jit(_rng)
    stop_time = datetime.now()
    print(f"Stage 2 Elapsed {stop_time-start_time}")

    __profile_dir = os.path.join(ROOT_DIR, 'mem_profile')
    os.makedirs(__profile_dir, exist_ok=True)
    jax.profiler.save_device_memory_profile(os.path.join(__profile_dir, 'mem_stage_2.prof'))

    total_update_steps = int(config["NUM_UPDATES"] * config["NUM_EPISODES"])
    for team_ix, team_metrics in s2_episode_metrics.items():
        if team_fcp_agents[team_ix]:
            p_ix, partner_metrics = 0, team_metrics[0]
            plt.plot(range(total_update_steps), partner_metrics[0], label=f"{team_ix}")
    plt.legend()
    plt.savefig(f"./out/{JOB_ID}/stage-2_loss_per_partner.png")
    plt.close()


    # rollout_env_spec = EnvSpec("overcooked", 1, {"layout" : overcooked_layouts["cramped_room"]})
    # rollout_teams = [ TeamSpec(make_ppo_agent, 1, ['agent_0', 'agent_1']), ]
    # rollout_team_fcp_agents = [make_ppo_agent, ]
    # rollout_load_step = config["_CHECKPOINT_STEPS"][-1]
    # rollout_state_seq = get_rollout(config, rollout_env_spec, rollout_teams, rollout_team_fcp_agents, rollout_load_step, max_steps=300)
    # env = jaxmarl.make(rollout_env_spec.env_id, **rollout_env_spec.env_kwargs)
    # viz =  OvercookedVisualizer()
    # viz.animate(rollout_state_seq, env.agent_view_size, filename=f'./out/{JOB_ID}/fcp-animation.gif')

    return None

    


if __name__ == "__main__":
    main()
