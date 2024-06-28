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


# TODO
class EnvSpec(NamedTuple):
    env_id: str
    count: int
    env_kwargs: Any

class TeamSpec(NamedTuple):
    agent_class: Type['SelfPlayAgent']
    agent_count: int
    agent_uids: list['AgentUID']

class AgentUID(NamedTuple):
    env_ix: int
    agent_id: str

class EnvMapping(NamedTuple):
    # List of environment id's that will be invoked along with how many environments of that id will be created
    envs: list[EnvSpec]
    # List of teams, each team is a list of tuples specifying the environment index in envs and the agend id
    # It is assumed that all agents in the same team have the same action space and observation space
    # And can therefore be interchanged without issue for ficticious coplay
    teams: list[TeamSpec]


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


    

class SelfPlayAgent:
    def __init__(self, rng, env_specs: list[EnvSpec], team_spec: TeamSpec, environments):
        pass

    def process_observation(self, obsv):
        """
        Generate the agent output from a given (vectorised) observation.
        This is the functional approach to carry over all relevant data
        to the update function that will be called later.
        E.g. if the agent uses and ActorCritic network with policy and value component,
        this function will return both.
        Use the get_action function to then extract the relevant action from the information
        that is returned here
        """
        raise NotImplementedError("This interface method needs to be subclassed")

    def get_action(self, processed_observation):
        raise NotImplementedError("This interface method needs to be subclassed")
    
    def update(self):
        raise NotImplementedError("This interface method needs to be subclassed")
    

def _make_envs_step(map_agent_uid_to_partner_instance: dict[int, dict[str, jax.Array], int], env_mapping: EnvMapping):
    def _envs_step(runner_state, _):
        env_stack, environments, partners, rng = runner_state

        transformed_observations = []
        transformed_actions = []
        for team_partners in partners:
            transformed_observations.append([ [] for p in team_partners ])
            transformed_actions.append([ [] for p in team_partners ])

        # Transform all observation vectors to group them by partner agents
        for env_ix, (obsv, _) in enumerate(env_stack):
            sorted_items = sorted(map_agent_uid_to_partner_instance[env_ix].items(), key=lambda t: t[0])
            for agent_id, (team_ix, masks, num_partner_instances) in sorted_items:
                for i in range(num_partner_instances):
                    transformed_observations[team_ix][i].append(obsv[agent_id][masks[i]])


        # Run each transformed stack of observations through their respective parntner agents
        for team_ix, team_partners in enumerate(partners):
            for p_ix, partner in enumerate(team_partners):
                stacked_obsv = jnp.concatenate(transformed_observations[team_ix][p_ix], axis=0)
                processed = partner.process_observation(stacked_obsv)
                stacked_actions = partner.get_action(processed)

                obsv_shapes = np.array([ a.shape[0] for a in transformed_observations[team_ix][p_ix] ])
                fragmented_actions = jnp.split(stacked_actions, np.cumsum(obsv_shapes))
                transformed_actions[team_ix][p_ix] = fragmented_actions


        # Reverse the process with the stacked actions to use them for stepping the environments
        action_stack = []
        for obsv, _ in env_stack:
            action_v = { k: jnp.zeros(v.shape[0], dtype=int) for k, v in obsv.items() }
            action_stack.append(action_v)
            
        for env_ix, (obsv, _) in enumerate(env_stack):
            sorted_items = sorted(map_agent_uid_to_partner_instance[env_ix].items(), key=lambda t: t[0])
            for agent_id, (team_ix, masks, num_partner_instances) in sorted_items:
                for i in range(num_partner_instances):
                    mask = masks[i]
                    action_v = action_stack[env_ix]
                    a_fragment = transformed_actions[team_ix][i].pop(0)
                    action_v[agent_id] = action_v[agent_id].at[mask].set(a_fragment)

        
        #STEP ENVIRONMENTS
        for env_ix, env in enumerate(environments):
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, env_mapping.envs[env_ix].count)
            env_state = env_stack[env_ix][1]
            env_act = action_stack[env_ix]
            obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0,0,0)
                )(rng_step, env_state, env_act)


    return _envs_step



def _update_env(runner_state: tuple[Any, Any, list[SelfPlayAgent], list[int], Any], _):
    # env_stack, environments, env_instance_counts, partners, rng = runner_state

    # # Select actions taken by all agents
    # for agent in partners:
    #     agent_response = agent.process_observation(obsv_batch)
    #     agent_actions = agent.get_action(agent_response)


    # # Step all environments
    # for env_ix, env in enumerate(environments):
    #     rng_step = jax.random.split(rng, env_instance_counts[env_ix])
    #     obsv, env_state, reward, done, info = jax.vmap(
    #         env.step, in_axes=(0,0,0)
    #     )(rng_step, env_state, env_act)

    # # collect transition information
    pass

def _update_step(runner_state, _):
    # Collect trajectories from all environments
    pass



def _make_stage_1(config, env_mapping: EnvMapping, numpy_seed: int):
    # Key Assumptions:
    # - Every agent in the environment has
    #   - the same action space
    #   - the same observation space

    # In practice, each environment in the env mapping is parallelised separately
    # since we have no guarantee that action/observation spaces all match up
    environments = []
    for _env in env_mapping.envs:
        environments.append(jaxmarl.make(_env.env_id))

    np_rng = np.random.default_rng(numpy_seed)

    def _stage_1(rng):
        # Create all parallelised environments
        env_stacks = []
        for env_ix, env in enumerate(environments):
            rng_array = jax.random.split(rng, env_mapping.envs[env_ix].count)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, ))(rng_array)
            env_stacks.append( (obsv, env_state) )

        # Create all parallelised partner agents for each team
        partners = []
        for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(env_mapping.teams):
            team_partners = []
            for _ in range(partner_count):
                rng_key, rng = jax.random.split(rng)
                team_partners.append(
                    cls_SelfPlayAgent(rng_key, env_mapping.envs, env_mapping.teams[team_ix], environments)
                )
            partners.append(team_partners)

        for episode_ix in range(config["NUM_EPISODES"]):
            # Randomly sample partners to assign to agents in their respective teams for each environment
            # Generate this using numpy so that jax can treat it as static values
            # Important for later splitting up the arrays
            map_agent_uid_to_partner_instance = dict()
            for (team_ix, team_spec) in enumerate(env_mapping.teams):
                for (env_ix, agent_id) in team_spec.agent_uids:
                    if env_ix not in map_agent_uid_to_partner_instance:
                        map_agent_uid_to_partner_instance[env_ix] = dict()
                    partner_count = team_spec.agent_count
                    env_instance_count = env_mapping.envs[env_ix].count
                    sampled_partners = np_rng.choice(partner_count, size=(env_instance_count, ))
                    masks = []
                    for i in range(partner_count):
                        masks.append(np.isclose(sampled_partners, i))
                    map_agent_uid_to_partner_instance[env_ix][agent_id] = (
                        env_ix,
                        masks,
                        partner_count
                    )
            

            # Simulate all agents in all environments
            # and collect the traversed states with associating transition information
            # last_runner_state, trajectories = jax.lax.scan(_update_step, TODO, None, length=config["ENV_STEPS"])
            # TODO
            _make_envs_step(map_agent_uid_to_partner_instance, env_mapping)((env_stacks, environments, partners, rng), None)

            break
        return None

    return _stage_1



class FCP:
    @staticmethod
    def make_stage_1(config, env_mapping: EnvMapping, numpy_seed:int):
        return _make_stage_1(config, env_mapping, numpy_seed)