from copy import deepcopy
from typing import NamedTuple, Sequence, Type

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


"""
IDEA
Go back to what is mentioned in the paper
agents are randomly assigned to environments
=> many more environments than agets
=> Instead of centering the process around environments, center the process around agents
=> stack environments in one big vmap and keep track of which agent is assigned to which environment
=> Simulate all environments in parallel (duh) and collect observations according to the respective agents
=> parallelize each agent evaluating their next action and value and everything in their own loop probably
    (since I can't really see how that can be further parallelized bc that's not how vmap works)
=> we'll have to figure out the update loop later
"""

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

config = {
    "NUM_ENVS": 3, # 200
    "NUM_AGENTS": 3, # 32
    # "NUM_STEPS": 25
}

def _stage_1_init_actors(rng, env, config):
    pass

def _stage_1_init_envs(rng, env, config):
    rng_array = jax.random.split(rng, config["NUM_ENVS"])
    obsv_stacked, env_state_stacked = jax.vmap(env.reset, in_axes=(0, ))(rng_array)
    return obsv_stacked, env_state_stacked

def _make_stage_1(config):
    # Key Assumptions:
    # - Every agent in the environment has
    #   - the same action space
    #   - the same observation space
    config_cpy = deepcopy(config)
    env: SimpleReferenceMPE = jaxmarl.make("MPE_simple_reference_v3")

    config_cpy["NUM_ACTORS"] = env.num_agents * config_cpy["NUM_ENVS"]

    def _stage_1(rng):
        rng_array = jax.random.split(rng, config["NUM_ENVS"])
        obsv_stacked, env_state_stacked = jax.vmap(env.reset, in_axes=(0, ))(rng_array)

        # Create a new model for every agent
        model = SimpleNetwork(env.observation_space(env.agents[0]).shape, env.action_space(env.agents[0]).n)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        agents = []
        for _ in range(config['NUM_AGENTS']):
            agents.append(model.init(rng, init_x))


        # Use random sampling to assign agents to environments
        for ix_env in range(config_cpy["NUM_ENVS"]):
            sampled_agents = jax.random.choice(rng, agents, (env.num_agents, ), replace=True)
            print(sampled_agents)
            break

    return _stage_1


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    stage_1_jit = jax.jit(_make_stage_1(config))
    stage_1_jit(rng)





    # This doesn't seem to work because models_stacked seems to return a dictionary
    # We need to keep the models separate it seems and then use vmap when evaluating
    # rng = jax.random.PRNGKey(0)
    # env: SimpleReferenceMPE = jaxmarl.make("MPE_simple_reference_v3")
    # rng_array = jax.random.split(rng, config["NUM_ENVS"])
    # obsv_stacked, env_state_stacked = jax.vmap(env.reset, in_axes=(0, ))(rng_array)

    # rng_array = jax.random.split(rng, config["NUM_AGENTS"])
    # obsv_space_shape = env.observation_space(env.agents[0]).shape
    # init_x_shape = tuple([config["NUM_AGENTS"]] + list(obsv_space_shape))
    # init_x = jnp.zeros(init_x_shape)
    # model = SimpleNetwork(env.observation_space(env.agents[0]).shape, env.action_space(env.agents[0]).n)
    # models_stacked = jax.vmap(model.init, in_axes=(0, 0))(rng_array, init_x)
    # print(models_stacked(obsv_stacked['agent_0']))

    # env_tuples = [ env.reset(_rng) for _rng in rng_array ]
    # obsv_stacked, env_state_stacked = [ t[0] for t in env_tuples ], [ t[1] for t in env_tuples ]

    
    # instantiate new network class based on action space and observation space
    # (which is assumed to be common)
    # use network.init with vmap to have many networks with unique initial seeds
    


    # one, two = jax.vmap(env.reset)(rng_array)
    # print(one)
    # print(two)
    # 

    # print(obsv_stacked)
