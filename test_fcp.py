from collections import OrderedDict
from copy import deepcopy
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

from ficticious_coplay.fcp import FCP, EnvMapping, EnvSpec, SelfPlayAgent, AgentUID, TeamSpec


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


class TestSimpleAgent(SelfPlayAgent):

    def __init__(self, rng, env_specs: list[EnvSpec], team_spec: TeamSpec, environments):
        super(TestSimpleAgent, self).__init__(rng, env_specs, team_spec, environments)
        sample_env = environments[team_spec.agent_uids[0].env_ix]
        sample_agent_id = team_spec.agent_uids[0].agent_id
        self.actions = sample_env.action_space(sample_agent_id)
        self.rng = rng

    def process_observation(self, obsv):
        count = obsv.shape[0]
        actions = []
        for _ in range(count):
            key, self.rng = jax.random.split(self.rng)
            actions.append(self.actions.sample(key))
        key, self.rng = jax.random.split(self.rng)
        return jnp.stack(actions), jax.random.normal(key, (count, ))

    def get_action(self, processed_observation):
        return processed_observation[0]
    
    def update(self):
        return None
    

config = {
    "ENV_STEPS": 25,
    "NUM_EPISODES": 3
    # "NUM_ENVS": 3, # 200
    # "NUM_AGENTS": 3, # 32
    # "NUM_STEPS": 25
}


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    # This is part of the config
    # All environments specified here must have the same action space and observation space dimensions
    env_mapping = EnvMapping(
        envs=[ EnvSpec("MPE_simple_reference_v3", 8, {}) ],
        teams=[
            TeamSpec(TestSimpleAgent, 3, [AgentUID(0, 'agent_0'), AgentUID(0, 'agent_1')]),
            ]
    )

    # stage_1_jit = FCP.make_stage_1(config, env_mapping)
    stage_1_jit = jax.jit(FCP.make_stage_1(config, env_mapping))
    stage_1_jit(rng)