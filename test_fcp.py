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

from ficticious_coplay.fcp import FCP, EnvMapping, EnvSpec, SelfPlayAgent, AgentUID, SelfPlayAgentFactory, TeamSpec, _make_stage_2


class SimpleNetwork(nn.Module):
    input_dim: Sequence[int] # observation_space_dim
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
    

def make_simple_agent(init_rng, config, env_specs: list[EnvSpec], team_spec: TeamSpec, environments):
    sample_env = environments[team_spec.agent_uids[0].env_ix]
    sample_agent_id = team_spec.agent_uids[0].agent_id
    available_actions = sample_env.action_space(sample_agent_id)

    def process_observation(rng, obsv):
        count = obsv.shape[0]
        actions = jnp.empty(shape=obsv.shape[0], dtype=int)
        for i in range(count):
            rng, key = jax.random.split(rng)
            actions = actions.at[i].set(available_actions.sample(key))
        rng, key = jax.random.split(rng)
        return actions, jax.random.normal(key, (count, ))
    
    def get_action(rng, processed_observation):
        return processed_observation[0]
    
    def update(rng, trajectories):
        return None
    
    return SelfPlayAgent(
        process_observation,
        get_action,
        update,
        {}
    )


# class TestSimpleAgentFactory(SelfPlayAgentFactory):

#     def __init__(self, config, env_specs: list[EnvSpec], team_spec: TeamSpec, environments):
#         super(TestSimpleAgentFactory, self).__init__(config, env_specs, team_spec, environments)
#         self.config = config

#         sample_env = environments[team_spec.agent_uids[0].env_ix]
#         sample_agent_id = team_spec.agent_uids[0].agent_id
#         self.actions = sample_env.action_space(sample_agent_id)

#     def make_fn_process_observation(self, init_rng):
#         available_actions = self.actions
#         def process_observation(rng, obsv):
#             count = obsv.shape[0]
#             actions = []
#             for _ in range(count):
#                 rng, key = jax.random.split(rng)
#                 actions.append(available_actions.sample(key))
#             rng, key = jax.random.split(rng)
#             return jnp.stack(actions), jax.random.normal(key, (count, ))
#         return process_observation
    
#     def make_fn_get_action(self, init_rng):
#         def get_action(rng, processed_observation):
#             return processed_observation[0]
#         return get_action
    
#     def make_fn_update(self, init_rng):
#         def update(rng, trajectories):
#             return None
#         return update
    

config = {
    "ENV_STEPS": 25,
    "NUM_UPDATES": 1,
    "NUM_EPISODES": 50
    # "NUM_ENVS": 3, # 200
    # "NUM_AGENTS": 3, # 32
    # "NUM_STEPS": 25
}

def main():
    rng = jax.random.PRNGKey(0)
    numpy_seed = 0

    # This is part of the config
    # All environments specified here must have the same action space and observation space dimensions
    env_mapping = EnvMapping(
        envs=[ EnvSpec("MPE_simple_reference_v3", 200, {}) ],
        teams=[
            TeamSpec(make_simple_agent, 8, [AgentUID(0, 'agent_0'), AgentUID(0, 'agent_1')]),
            ]
    )

    rng, _rng = jax.random.split(rng)
    stage_1_jit = FCP.make_stage_1(config, env_mapping, numpy_seed)
    # stage_1_jit = jax.jit(FCP.make_stage_1(config, env_mapping, numpy_seed))
    episode_metrics, partners = stage_1_jit(_rng)

    team_fcp_agents = [make_simple_agent, ]
    rng, _rng = jax.random.split(rng)
    stage_2_jit = _make_stage_2(config, env_mapping, partners, team_fcp_agents, numpy_seed)
    stage_2_jit(_rng)


if __name__ == "__main__":
    start_time = datetime.now()
    jax.jit(main)()
    stop_time = datetime.now()
    print(f"Elapsed {stop_time-start_time}")