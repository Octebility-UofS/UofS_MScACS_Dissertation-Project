from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import Any, NamedTuple, Sequence, Type

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.environments import SimpleReferenceMPE
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import optax

from ficticious_coplay.fcp import FCP, EnvMapping, EnvSpec, SelfPlayAgent, AgentUID, TeamSpec, _make_stage_2, get_rollout


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
    


def make_ppo_agent(init_rng, config, env_spec: EnvSpec, team_spec: TeamSpec, env):
    network = SimpleNetwork(env.action_space().n)
    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()
    init_rng, _rng = jax.random.split(init_rng)
    network_params = network.init(_rng, init_x)
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

    init_agent_state = {}
    init_agent_state["eval"] = False
    init_agent_state["train_state"] = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    def get_action(rng, obsv, flattened_obsv, agent_state):
        pi, value = network.apply(agent_state["train_state"].params, flattened_obsv)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        return agent_state, action, (pi, value, log_prob)
    

    # This actually seems to be IPPO loss
    # TODO
    def _loss_fn(net_train_state_params, traj_batch, gae, targets):
        # net_train_state_params = agent_state["train_state"].params
        # gae = Advantages
        # RERUN NETWORK
        flattened_obs = traj_batch.obs.reshape((traj_batch.obs.shape[0], traj_batch.obs.shape[1], -1))
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

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, gradients = grad_fn(agent_state["train_state"].params, trajectories, advantages, targets)
        agent_state["train_state"] = agent_state["train_state"].apply_gradients(grads=gradients)

        return agent_state, total_loss
    
    return SelfPlayAgent(
        get_action,
        update
    ), init_agent_state
    

# def make_simple_agent(init_rng, config, env_spec: EnvSpec, team_spec: TeamSpec, env):
#     sample_agent_id = team_spec.agent_uids[0]
#     available_actions = env.action_space(sample_agent_id)
#     init_agent_state = {}

#     def get_action(rng, obsv, flattened_obsv, agent_state):
#         count = obsv.shape[0]
#         actions = jnp.empty(shape=obsv.shape[0], dtype=jnp.int32)
#         for i in range(count):
#             rng, key = jax.random.split(rng)
#             actions = actions.at[i].set(available_actions.sample(key))
#         rng, key = jax.random.split(rng)
#         return agent_state, actions, (jax.random.normal(key, (count, )), )
    
#     def update(rng, trajectories, agent_state):
#         return agent_state, None
    
#     return SelfPlayAgent(
#         get_action,
#         update
#     ), init_agent_state
    

config = {
    "ENV_STEPS": 1e6,
    "NUM_UPDATES": 100,
    "NUM_EPISODES": 6,
    # "ANNEAL_LR": True,
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

def main():
    rng = jax.random.PRNGKey(0)
    numpy_seed = 0

    # This is part of the config
    # All environments specified here must have the same action space and observation space dimensions
    env_spec = EnvSpec("overcooked", 200, {"layout" : overcooked_layouts["cramped_room"]})
    teams = [ TeamSpec(make_ppo_agent, 8, ['agent_0', 'agent_1']), ]

    rng, _rng = jax.random.split(rng)
    stage_1_jit = FCP.make_stage_1(config, env_spec, teams, numpy_seed)
    # stage_1_jit = jax.jit(FCP.make_stage_1(config, env_mapping, numpy_seed))
    episode_metrics, partners, last_episode_runner_state = stage_1_jit(_rng)

    env_obsv_state, partner_states, rng = last_episode_runner_state

    single_partner_states = []
    for team_partners in partner_states:
        single_partner_states.append([team_partners[0], ])

    env_spec = EnvSpec("overcooked", 1, {"layout" : overcooked_layouts["cramped_room"]})
    teams = [ TeamSpec(make_ppo_agent, 1, ['agent_0', 'agent_1']), ]
    state_seq = get_rollout(config, single_partner_states, env_spec, teams, max_steps=30)

    return state_seq

    team_fcp_agents = [make_ppo_agent, ]
    rng, _rng = jax.random.split(rng)
    stage_2_jit = _make_stage_2(config, env_spec, teams, partners, team_fcp_agents, numpy_seed)
    stage_2_jit(_rng)


if __name__ == "__main__":
    start_time = datetime.now()
    state_seq = jax.jit(main)()
    stop_time = datetime.now()
    print(f"Elapsed {stop_time-start_time}")

    env = jaxmarl.make("overcooked", **{"layout" : overcooked_layouts["cramped_room"]})

    viz =  OvercookedVisualizer()
    viz.animate(state_seq, env.agent_view_size, filename='animation.gif')