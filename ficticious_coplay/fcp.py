from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, NamedTuple, Sequence, Type

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
    agent_class: Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any], 'SelfPlayAgent']
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

class SelfPlayAgent(NamedTuple):
    process_observation: Callable[[jnp.ndarray], Any] # obs_v -> processed_observation
    get_action: Callable[[Any], jnp.ndarray] # processed_observation -> actions_v
    update: Callable[[Any, Any, Any], Any] # config, rng, trajectories -> metrics
    attrs: dict[str, Any] # Dict of object attributes, this helps keep all functions pure
    

class SelfPlayAgentFactory:
    def __init__(self, config, env_specs: list[EnvSpec], team_spec: TeamSpec, environments):
        self.attrs = {"eval": False}

    def create(self, init_rng):
        agent = SelfPlayAgent(
            self.make_fn_process_observation(init_rng),
            self.make_fn_get_action(init_rng),
            self.make_fn_update(init_rng),
            self.attrs
        )
        return agent
    
    def make_fn_process_observation(self, init_rng):
        def process_observation(obsv):
            raise NotImplementedError("This interface method needs to be subclassed")
        return process_observation
    
    def make_fn_get_action(self, init_rng):
        def get_action(processed_observation):
            raise NotImplementedError("This interface method needs to be subclassed")
        return get_action
    
    def make_fn_update(self, init_rng):
        def update(config, rng, trajectories):
            raise NotImplementedError("This interface method needs to be subclassed")
        return update
    

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    processed_observation: Any
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    

def _make_envs_step(map_agent_uid_to_partner_instance: dict[int, dict[str, jax.Array], int], env_mapping: EnvMapping, environments, partners):
    def _envs_step(runner_state, _):
        env_stack, rng = runner_state

        transformed_observations = []
        transformed_actions = []
        transformed_stacked_observations = [] # Stacked (batched) tensor of observations per agent
        transformed_stacked_actions = [] # Stacked (batched) tensor of actions per agent
        transformed_stacked_processed_agent_osbv = [] # Whatever is returned by an agent when an observation is passed
        for team_partners in partners:
            transformed_observations.append([ [] for p in team_partners ])
            transformed_actions.append([ [] for p in team_partners ])
            transformed_stacked_observations.append([ [] for p in team_partners ])
            transformed_stacked_actions.append([ [] for p in team_partners ])
            transformed_stacked_processed_agent_osbv.append([ [] for p in team_partners ])

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
                rng, _rng = jax.random.split(rng)
                processed = partner.process_observation(_rng, stacked_obsv)
                rng, _rng = jax.random.split(rng)
                stacked_actions = partner.get_action(_rng, processed)

                obsv_shapes = np.array([ a.shape[0] for a in transformed_observations[team_ix][p_ix] ])
                fragmented_actions = jnp.split(stacked_actions, np.cumsum(obsv_shapes))
                transformed_actions[team_ix][p_ix] = fragmented_actions

                # Saved for creating transitions later on
                # Avoid additional re-computation while trading off increased memory requirements
                transformed_stacked_observations[team_ix][p_ix] = stacked_obsv
                transformed_stacked_actions[team_ix][p_ix] = stacked_actions
                transformed_stacked_processed_agent_osbv[team_ix][p_ix] = processed


        # Reverse the process with the stacked actions to use them for stepping the environments
        action_stack = []
        for obsv, _ in env_stack:
            action_v = { k: jnp.zeros(v.shape[0], dtype=jnp.int32) for k, v in obsv.items() }
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
        updated_env_stack = []
        env_step_data = []
        for env_ix, env in enumerate(environments):
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, env_mapping.envs[env_ix].count)
            env_state = env_stack[env_ix][1]
            env_act = action_stack[env_ix]
            obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0,0,0)
                )(rng_step, env_state, env_act)
            updated_env_stack.append( (obsv, env_state) )
            env_step_data.append( (obsv, env_state, reward, done, info) )

        # COLLECT TRANSITIONS FOR EACH AGENT
        # Will be later passed to update functions for each agent
        transformed_reward = []
        transformed_done = []
        for team_partners in partners:
            transformed_reward.append([ [] for p in team_partners ])
            transformed_done.append([ [] for p in team_partners ])
        transformed_stacked_reward = []
        transformed_stacked_done = []
        # info is different because it is on a per-environment basis
        # TODO figure out how to deal with this (not very clear in documentation)
        # transformed_stacked_info = []
        for env_ix, (obsv, _) in enumerate(env_stack):
            _, _, reward, done, info = env_step_data[env_ix]
            sorted_items = sorted(map_agent_uid_to_partner_instance[env_ix].items(), key=lambda t: t[0])
            for agent_id, (team_ix, masks, num_partner_instances) in sorted_items:
                for i in range(num_partner_instances):
                    transformed_reward[team_ix][i].append(reward[agent_id][masks[i]])
                    transformed_done[team_ix][i].append(done[agent_id][masks[i]])
                    # TODO
                    # transformed_stacked_info[team_ix][i].append(info[agent_id][masks[i]])
        # Concatenate reward and done fragmented arrays
        for team_ix in range(len(transformed_reward)):
            team_stacked_reward = []
            team_stacked_done = []
            for i in range(len(transformed_reward[team_ix])):
                team_stacked_reward.append(jnp.concatenate(transformed_reward[team_ix][i], axis=0))
                team_stacked_done.append(jnp.concatenate(transformed_done[team_ix][i], axis=0))
            transformed_stacked_reward.append(team_stacked_reward)
            transformed_stacked_done.append(team_stacked_done)

        agent_transitions = []
        for team_ix in range(len(partners)):
            agent_transitions.append([])
            for p_ix in range(len(partners[team_ix])):
                agent_transitions[team_ix].append(Transition(
                    transformed_stacked_done[team_ix][p_ix],
                    transformed_stacked_actions[team_ix][p_ix],
                    transformed_stacked_processed_agent_osbv[team_ix][p_ix],
                    transformed_stacked_reward[team_ix][p_ix],
                    transformed_stacked_observations[team_ix][p_ix],
                    {} # TODO
                ))


        
        runner_state = updated_env_stack, rng
        return runner_state, agent_transitions


    return _envs_step


def _make_update_step(config, map_agent_uid_to_partner_instance, env_mapping: EnvMapping, environments, partners: list[list[SelfPlayAgent]]):
    def _update_step(runner_state: tuple[Any, Any, list[SelfPlayAgent], list[int], Any], _):
        # Collect Trajectories
        runner_state, trajectories = jax.lax.scan(
            _make_envs_step(map_agent_uid_to_partner_instance, env_mapping, environments, partners),
            runner_state, None,
            length=config["ENV_STEPS"]
        )


        # Update each agent given their trajectories
        metrics = []
        env_stack, rng = runner_state
        for team_ix, team_partners in enumerate(partners):
            team_metrics = []
            for p_ix, partner in enumerate(team_partners):
                rng, _rng = jax.random.split(rng)
                update_metrics = partner.update(_rng, trajectories[team_ix][p_ix])
                team_metrics.append(update_metrics)
            metrics.append(team_metrics)

        runner_state = env_stack, rng
        return runner_state, metrics
    return _update_step


def _make_episode(config, env_mapping: EnvMapping, environments, partners, map_agent_uid_to_partner_instance):
    def _episode(runner_episode, _):
        env_stacks, rng = runner_episode

        rng, _rng = jax.random.split(rng)

        runner_state = env_stacks, _rng
        last_runner_state, metrics = jax.lax.scan(
                _make_update_step(config, map_agent_uid_to_partner_instance, env_mapping, environments, partners),
                runner_state, None,
                length=config["NUM_UPDATES"]
            )
        env_stacks, rng = last_runner_state

        runner_episode = env_stacks, rng
        return runner_episode, metrics
    
    return _episode



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
                rng, _rng = jax.random.split(rng)
                team_partners.append(
                    cls_SelfPlayAgent(_rng, config, env_mapping.envs, env_mapping.teams[team_ix], environments)
                )
            partners.append(team_partners)


        # We need to randomly sample partner agents to assign them to environment instances
        # The current assumption is that this random sampling is the same for each episode
        # Boolean masks are used to be able to split them up during stepping and combine them for each partner instance
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

        # Scan over episodes
        episode_runner_state = env_stacks, rng
        last_episode_runner_state, episode_metrics = jax.lax.scan(
            _make_episode(config, env_mapping, environments, partners, map_agent_uid_to_partner_instance),
            episode_runner_state, None,
            length=config["NUM_EPISODES"]
        )

        return episode_metrics, partners

    return _stage_1



def _make_stage_2(
    config,
    env_mapping: EnvMapping,
    partners: list[list[SelfPlayAgent]],
    cls_team_actors: list[Type[SelfPlayAgent]],
    numpy_seed: int
    ):
    environments = []
    for _env in env_mapping.envs:
        environments.append(jaxmarl.make(_env.env_id))

    np_rng = np.random.default_rng(numpy_seed)

    def _stage_2(rng):
        # Create all parallelised environments
        env_stacks = []
        for env_ix, env in enumerate(environments):
            rng_array = jax.random.split(rng, env_mapping.envs[env_ix].count)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0, ))(rng_array)
            env_stacks.append( (obsv, env_state) )

        # Add the fcp agent of each team to a modified list of partner agents
        modified_partners = []
        for team_ix, team_partners in enumerate(partners):
            rng, _rng = jax.random.split(rng)
            modified_partners.append(
                [ cls_team_actors[team_ix](_rng, config, env_mapping.envs, env_mapping.teams[team_ix], environments) ]
                + team_partners
            )


        # Randomly sample partners to assign to agents in their respective teams for each environment
        # Generate this using numpy so that jax can treat it as static values
        # Important for later splitting up the arrays
        # ====
        # Key difference to Stage 1 mapping is that all 'partners' are bumped by one index value (+1)
        # Each environment will only have fcp agent (the one to be trained) 
        # This will be sampled randomly (which requires a large number of environments to make good training progress)

        # For each environment instance, randomly select one agent that will be the fcp target agent
        fcp_agent_map = dict()
        for env_ix, env in enumerate(environments):
            target_agents = [ env.agents[_i] for _i in np_rng.choice(env.num_agents, env_mapping.envs[env_ix].count) ]
            fcp_agent_map[env_ix] = target_agents

        map_agent_uid_to_partner_instance = dict()
        for (team_ix, team_partners) in enumerate(modified_partners):
            for (env_ix, agent_id) in env_mapping.teams[env_ix].agent_uids:
                if env_ix not in map_agent_uid_to_partner_instance:
                    map_agent_uid_to_partner_instance[env_ix] = dict()
                partner_count = len(team_partners)
                env_instance_count = env_mapping.envs[env_ix].count
                # Add +1 after sampling because we want to ignore the target fcp agent at index 0
                # Choice range is -1 actual to account for that
                sampled_partners = np_rng.choice(partner_count-1, size=(env_instance_count, ))+1
                # Replace the 'sampled_partners' with 0 index according to 'fcp_agent_map'
                for _i in range(sampled_partners.shape[0]):
                    if fcp_agent_map[env_ix][_i] == agent_id:
                        sampled_partners[_i] = 0
                # Continue as with stage 1 with masking
                masks = []
                for i in range(partner_count):
                    masks.append(np.isclose(sampled_partners, i))
                map_agent_uid_to_partner_instance[env_ix][agent_id] = (
                    env_ix,
                    masks,
                    partner_count
                )

         # Scan over episodes
        episode_runner_state = env_stacks, rng
        last_episode_runner_state, episode_metrics = jax.lax.scan(
            _make_episode(config, env_mapping, environments, modified_partners, map_agent_uid_to_partner_instance),
            episode_runner_state, None,
            length=config["NUM_EPISODES"]
        )
        
        return episode_metrics, partners

    return _stage_2



class FCP:
    @staticmethod
    def make_stage_1(config, env_mapping: EnvMapping, numpy_seed:int):
        return _make_stage_1(config, env_mapping, numpy_seed)

    @staticmethod
    def make_stage_2(config, partners: list[list[SelfPlayAgent]], cls_team_actors: list[Type[SelfPlayAgent]]):
        return _make_stage_2(config)