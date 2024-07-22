from collections import OrderedDict
from copy import deepcopy
from functools import partial
import os
import pickle
from typing import Any, Callable, NamedTuple, Sequence, Type

import flax.linen as nn
import jax
import jax.experimental
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.environments import SimpleReferenceMPE
import orbax.checkpoint as ocp
from flax.training import orbax_utils

from ficticious_coplay.util.util import rec_frozenset


# TODO
class EnvSpec(NamedTuple):
    env_id: str
    count: int
    env_kwargs: Any

class TeamSpec(NamedTuple):
    agent_class: Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any, Any], 'SelfPlayAgent']
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
    get_action: Callable[[Any, jnp.ndarray, jnp.ndarray, Any], tuple[Any, jnp.ndarray, Any]] # rng, obs_v, flattened_obs_v, agent_state -> update_agent_state, actions_v, aux_data
    update: Callable[[Any, Any, Any], tuple[Any, Any]] # config, rng, trajectories -> updated_agent_state, metrics
    save: Callable[[Any, int], None] # agent_state, step -> None
    load: Callable[[Any, int], Any] # agent_state, step -> new_agent_state
    

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    processed_observation: Any
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@partial(jax.jit, static_argnames=['map_agent_uid_to_partner_instance',])
def transform_env_partner(
    pytree,
    map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[int, tuple[tuple[str, tuple[int, int]], ...]]]]]
    ):
    transformed_mapping = dict()
    for team_ix, partner_mapping in map_agent_uid_to_partner_instance:
        transformed_mapping[team_ix] = dict()
        for partner_ix, agent_mapping in partner_mapping:
            transformed_mapping[team_ix][partner_ix] = jnp.concatenate(
                [ pytree[agent_id][s0: s1] for agent_id, (s0, s1) in agent_mapping ]
            )
    return transformed_mapping

@partial(jax.jit, static_argnames=['reverse_map_agent_uid_to_partner_instance',])
def untransform_env_partner(
    transformed_mapping,
    reverse_map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[str, tuple[tuple[tuple[int, int], tuple[int, tuple[int, int]]], ...]]]]]
    ):
    pytree = dict()
    for team_ix, agent_mapping in reverse_map_agent_uid_to_partner_instance:
        for agent_id, partner_mapping in agent_mapping:
            # We don't need to worry about other teams as we explictly assign each agent to exactly one team
            pytree[agent_id] = jnp.concatenate(
                [ transformed_mapping[team_ix][partner_ix][s0: s1] for _, (partner_ix, (s0, s1)) in partner_mapping ]
            )
    return pytree

    

def _make_envs_step(
    map_agent_uid_to_partner_instance: dict[int, dict[int, tuple[int, int]]],
    env_spec: EnvSpec, env,
    partners: list[list[SelfPlayAgent]]
    ):
    def _envs_step(runner_state, _):
        env_obsv_state, partner_states, rng = runner_state

        transformed_observations = []
        transformed_actions = []
        transformed_stacked_observations = [] # Stacked (batched) tensor of observations per agent
        transformed_stacked_actions = [] # Stacked (batched) tensor of actions per agent
        transformed_stacked_agent_aux_data = [] # Whatever is returned by an agent when an observation is passed
        for team_partners in partners:
            transformed_observations.append([ [] for p in team_partners ])
            transformed_actions.append([ [] for p in team_partners ])
            transformed_stacked_observations.append([ [] for p in team_partners ])
            transformed_stacked_actions.append([ [] for p in team_partners ])
            transformed_stacked_agent_aux_data.append([ [] for p in team_partners ])

        # Transform all observation vectors to group them by partner agents
        env_obsv, env_state = env_obsv_state
        sorted_items = sorted(map_agent_uid_to_partner_instance.items(), key=lambda t: t[0])
        for agent_id, (team_ix, masks, num_partner_instances) in sorted_items:
            for i in range(num_partner_instances):
                transformed_observations[team_ix][i].append(env_obsv[agent_id][masks[i]])


        # Run each transformed stack of observations through their respective parntner agents
        updated_partner_states = []
        for team_ix, team_partners in enumerate(partners):
            team_updated_partner_states = []
            for p_ix, partner in enumerate(team_partners):
                agent_state = partner_states[team_ix][p_ix]
                stacked_obsv = jnp.concatenate(transformed_observations[team_ix][p_ix], axis=0)
                flattened_stacked_obsv = stacked_obsv.reshape((stacked_obsv.shape[0], -1))

                rng, _rng = jax.random.split(rng)
                agent_state, stacked_actions, agent_aux_data = partner.get_action(_rng, stacked_obsv, flattened_stacked_obsv, agent_state)

                team_updated_partner_states.append(agent_state)

                obsv_shapes = np.array([ a.shape[0] for a in transformed_observations[team_ix][p_ix] ])
                fragmented_actions = jnp.split(stacked_actions, np.cumsum(obsv_shapes))
                transformed_actions[team_ix][p_ix] = fragmented_actions

                # Saved for creating transitions later on
                # Avoid additional re-computation while trading off increased memory requirements
                transformed_stacked_observations[team_ix][p_ix] = stacked_obsv
                transformed_stacked_actions[team_ix][p_ix] = stacked_actions
                transformed_stacked_agent_aux_data[team_ix][p_ix] = agent_aux_data
            updated_partner_states.append(team_updated_partner_states)


        # Reverse the process with the stacked actions to use them for stepping the environments
        env_obsv, env_state = env_obsv_state
        action_v = { k: jnp.zeros(v.shape[0], dtype=jnp.int32) for k, v in env_obsv.items() }

        sorted_items = sorted(map_agent_uid_to_partner_instance.items(), key=lambda t: t[0])
        for agent_id, (team_ix, masks, num_partner_instances) in sorted_items:
            for i in range(num_partner_instances):
                mask = masks[i]
                a_fragment = transformed_actions[team_ix][i].pop(0)
                action_v[agent_id] = action_v[agent_id].at[mask].set(a_fragment)
        env_act = action_v

        
        #STEP ENVIRONMENTS
        updated_env_stack = []
        env_step_data = []
        env_obsv, env_state = env_obsv_state
        env_act = env_act
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, env_spec.count)
        obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0,0,0)
            )(rng_step, env_state, env_act)
        updated_env_obsv_state = obsv, env_state
        env_step_data = (obsv, env_state, reward, done, info)

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
        sorted_items = sorted(map_agent_uid_to_partner_instance.items(), key=lambda t: t[0])
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
                    transformed_stacked_agent_aux_data[team_ix][p_ix],
                    transformed_stacked_reward[team_ix][p_ix],
                    transformed_stacked_observations[team_ix][p_ix],
                    {} # TODO Info info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                ))


        
        runner_state = updated_env_obsv_state, updated_partner_states, rng
        return runner_state, (agent_transitions, env_state)


    return _envs_step


def _make_update_step(
    config, map_agent_uid_to_partner_instance: dict[int, dict[int, tuple[int, int]]],
    env_spec: EnvSpec, teams: list[TeamSpec], env,
    partners: list[list[SelfPlayAgent]]
    ):
    def _update_step(runner_state: tuple[Any, Any, list[SelfPlayAgent], list[int], Any], save_counter):
        # Collect Trajectories
        runner_state, (trajectories, _) = jax.lax.scan(
            _make_envs_step(map_agent_uid_to_partner_instance, env_spec, env, partners),
            runner_state, None,
            length=config["ENV_STEPS"]
        )


        # Update each agent given their trajectories
        metrics = {}
        env_obsv_state, partner_states, rng = runner_state
        updated_partner_states = []
        for team_ix, team_partners in enumerate(partners):
            metrics[team_ix] = {}
            team_updated_partner_states = []
            for p_ix, partner in enumerate(team_partners):
                rng, _rng = jax.random.split(rng)
                updated_partner_state, update_metrics = partner.update(_rng, trajectories[team_ix][p_ix], partner_states[team_ix][p_ix])
                metrics[team_ix][p_ix] = update_metrics
                team_updated_partner_states.append(updated_partner_state)
            updated_partner_states.append(team_updated_partner_states)

        # After update, save agent weights as checkpoints
        for team_ix, team_partners in enumerate(partners):
            for p_ix, partner in enumerate(team_partners):
                agent_state = updated_partner_states[0][0]
                jax.experimental.io_callback(partner.save, agent_state, agent_state, save_counter)
                # jax.experimental.io_callback(_save_checkpoint, agent_state, config, partner, agent_state, save_counter)

        runner_state = env_obsv_state, updated_partner_states, rng
        return runner_state, (metrics, )
    return _update_step


def _make_episode(config, env_spec: EnvSpec, teams: list[TeamSpec], env, partners, map_agent_uid_to_partner_instance: dict[int, dict[int, tuple[int, int]]]):
    def _episode(runner_episode, counter):
        env_obsv_state, partner_states, rng = runner_episode

        rng, _rng = jax.random.split(rng)

        runner_state = env_obsv_state, partner_states, _rng
        last_runner_state, (metrics, ) = jax.lax.scan(
                _make_update_step(config, map_agent_uid_to_partner_instance, env_spec, teams, env, partners),
                runner_state,
                counter*config["NUM_UPDATES"] + jnp.arange(0, config["NUM_UPDATES"]), # Keep track of individual update step counter
                length=config["NUM_UPDATES"]
            )
        env_obsv_state, partner_states, rng = last_runner_state

        runner_episode = env_obsv_state, partner_states, rng
        return runner_episode, (metrics, )
    
    return _episode


def get_rollout(config, env_spec: EnvSpec, teams: list[TeamSpec], team_fcp_agents, rollout_load_step, max_steps=200):
    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)
    max_steps = min(max_steps, env.max_steps)

    np_rng = np.random.default_rng(0)
    rng = jax.random.PRNGKey(0)

    partners = []
    partner_states = []
    for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(teams):
        team_partners = []
        team_partner_states = []
        if team_fcp_agents[team_ix]:
            rng, _rng = jax.random.split(rng)
            fcp_prefix = f"fcp-{team_ix}_"
            partner, init_partner_state = team_fcp_agents[team_ix](
                _rng, config, env_spec, teams[team_ix], env,
                fcp_prefix
                )
            loaded_partner_state, _, _ = partner.load(init_partner_state, rollout_load_step)
            team_partner_states.append(loaded_partner_state)
            team_partners.append(partner)

        rng, _rng = jax.random.split(rng)
        checkpoint_prefix = f"{team_ix}-{0}_"
        partner, init_partner_state = cls_SelfPlayAgent(
            _rng, config, env_spec, teams[team_ix], env,
            checkpoint_prefix
            )
        loaded_partner_state, _, _ = partner.load(init_partner_state, rollout_load_step)
        team_partner_states.append(loaded_partner_state)
        team_partners.append(partner)
        partners.append(team_partners)
        partner_states.append(team_partner_states)

    map_agent_uid_to_partner_instance = dict()
    for (team_ix, team_spec) in enumerate(teams):
        for agent_ix, agent_id in enumerate(team_spec.agent_uids):
            if team_fcp_agents[team_ix]:
                partner_count = 2
                env_instance_count = 1
                if agent_ix == 0:
                    sampled_partners = np.array([0, ])
                    masks = []
                    for i in range(partner_count):
                        masks.append(np.isclose(sampled_partners, i))
                    map_agent_uid_to_partner_instance[agent_id] = ( team_ix, masks, partner_count )
                else:
                    sampled_partners = np.array([1, ])
                    masks = []
                    for i in range(partner_count):
                        masks.append(np.isclose(sampled_partners, i))
                    map_agent_uid_to_partner_instance[agent_id] = ( team_ix, masks, partner_count )
            else:
                partner_count = 1
                env_instance_count = 1
                sampled_partners = np.array([0, ])
                masks = []
                for i in range(partner_count):
                    masks.append(np.isclose(sampled_partners, i))
                map_agent_uid_to_partner_instance[agent_id] = ( team_ix, masks, partner_count )

    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, 1)
    obs, state = jax.vmap(env.reset, in_axes=(0, ))(rng_reset)

    state_seq = [state, ]
    runner_state = (obs, state), partner_states, rng
    _, (_, env_states) = jax.lax.scan(
            _make_envs_step(map_agent_uid_to_partner_instance, env_spec, env, partners),
            runner_state, None,
            length=max_steps
        )
    # Unbatchify the states since they're currently of shape [1, :]
    state_seq += [ jax.tree_map(lambda x: x[i], env_states) for i in range(max_steps) ]
    unbatched_state_seq = [ jax.tree_map(lambda x: x[0], state) for state in state_seq ]

    return unbatched_state_seq




def _make_stage_1(config, env_spec: EnvSpec, teams: list[TeamSpec], numpy_seed: int):

    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)

    np_rng = np.random.default_rng(numpy_seed)

    def _stage_1(rng):
        # Create all parallelised environments
        rng_array = jax.random.split(rng, env_spec.count)
        env_obsv_state = jax.vmap(env.reset, in_axes=(0, ))(rng_array)

        # Create all parallelised partner agents for each team
        partners = []
        partner_states = []
        for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(teams):
            team_partners = []
            team_partner_states = []
            for p_ix in range(partner_count):
                rng, _rng = jax.random.split(rng)
                checkpoint_prefix = f"{team_ix}-{p_ix}_"
                partner, partner_state = cls_SelfPlayAgent(
                    _rng, config,
                    env_spec, teams[team_ix], env,
                    checkpoint_prefix
                    )
                team_partners.append(partner)
                team_partner_states.append(partner_state)
            partners.append(team_partners)
            partner_states.append(team_partner_states)

        # Split the environment instances up into evenly sized ranges
        # For each agent in each team, ranges are randomly assigned for each partner instance
        # This vastly simplifies the complexity of transforming observations
        # While still keeping an element of randomness
        map_agent_uid_to_partner_instance = dict()
        for (team_ix, team_spec) in enumerate(teams):
            map_agent_uid_to_partner_instance[team_ix] = dict()
            for agent_id in team_spec.agent_uids:
                partner_count = team_spec.agent_count
                env_instance_count = env_spec.count
                # get evenly spaced 'slicing indexes'
                # according to number of partners and environments
                slice_ixs = np.linspace(0, env_instance_count, partner_count+1, dtype=int)
                # get a random permutation of assigning partner instances to slices
                partner_instance_permutation = np_rng.permutation(np.arange(partner_count))
                for ix, partner_ix in enumerate(partner_instance_permutation):
                    if partner_ix not in map_agent_uid_to_partner_instance[team_ix]:
                        map_agent_uid_to_partner_instance[team_ix][partner_ix] = dict()
                    map_agent_uid_to_partner_instance[team_ix][partner_ix][agent_id] = (slice_ixs[ix], slice_ixs[ix+1])
            # Convert dict of agent id's to slices into a list sorted by agent id
            for partner_ix, agent_mapping in map_agent_uid_to_partner_instance[team_ix].items():
                map_agent_uid_to_partner_instance[team_ix][partner_ix] = [
                    item for item in sorted(agent_mapping.items(), key=lambda t: t[0])
                ]
        
        print(map_agent_uid_to_partner_instance)

        reverse_map_agent_uid_to_partner_instance = dict()
        for (team_ix, team_spec) in enumerate(teams):
            reverse_map_agent_uid_to_partner_instance[team_ix] = dict()
            for partner_ix, agent_mapping in map_agent_uid_to_partner_instance[team_ix].items():
                slice_indexes = np.cumsum([0,] + [ s1 - s0 for agent_id, (s0, s1) in agent_mapping ])
                for ix, (agent_id, tuple_slice) in enumerate(agent_mapping):
                    if agent_id not in reverse_map_agent_uid_to_partner_instance[team_ix]:
                        reverse_map_agent_uid_to_partner_instance[team_ix][agent_id] = dict()
                    reverse_map_agent_uid_to_partner_instance[team_ix][agent_id][tuple_slice] = (partner_ix, (slice_indexes[ix], slice_indexes[ix+1]))
            # Convert dict of target tuple slices to a list in increasing order for correct concatenation
            for agent_id, tuple_mapping in reverse_map_agent_uid_to_partner_instance[team_ix].items():
                reverse_map_agent_uid_to_partner_instance[team_ix][agent_id] = [
                    item for item in sorted(tuple_mapping.items(), key=lambda t: t[0][0])
                ]

        print(reverse_map_agent_uid_to_partner_instance)

        map_agent_uid_to_partner_instance = rec_frozenset(map_agent_uid_to_partner_instance)
        reverse_map_agent_uid_to_partner_instance = rec_frozenset(reverse_map_agent_uid_to_partner_instance)

        test_tree = {'agent_0': jnp.arange(20), 'agent_1': 20+jnp.arange(20)}
        print(test_tree)

        transformed_tree = transform_env_partner(test_tree, map_agent_uid_to_partner_instance)
        print(transformed_tree)
        untransformed_tree = untransform_env_partner(transformed_tree, reverse_map_agent_uid_to_partner_instance)
        print(untransformed_tree)
        print("untransform inverse of transform?", jax.tree_util.tree_all(jax.tree_map(jnp.allclose, test_tree, untransformed_tree)))
        raise ValueError("Hi There :)")

        # Scan over episodes
        episode_runner_state = env_obsv_state, partner_states, rng
        last_episode_runner_state, (episode_metrics, ) = jax.lax.scan(
            _make_episode(config, env_spec, teams, env, partners, map_agent_uid_to_partner_instance),
            episode_runner_state,
            jnp.arange(0, config["NUM_EPISODES"]), # Used to keep track of saved network parameters
            length=config["NUM_EPISODES"]
        )

        unravelled_episode_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(jnp.concatenate([ x[i] for i in range(x.shape[0]) ]), axis=-1), episode_metrics)

        return unravelled_episode_metrics, last_episode_runner_state

    return _stage_1



def _make_stage_2(
    config,
    env_spec: EnvSpec, teams: list[TeamSpec],
    cls_team_fcp_agents: list[None|Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any, Any, Any], 'SelfPlayAgent']],
    checkpoint_load_steps: list[int],
    numpy_seed: int
    ):

    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)

    np_rng = np.random.default_rng(numpy_seed)

    def _stage_2(rng):
       # Create all parallelised environments
        rng_array = jax.random.split(rng, env_spec.count)
        env_obsv_state = jax.vmap(env.reset, in_axes=(0, ))(rng_array)


        # TODO Add more complex metrics to select checkpoints
        # Load 3 checkpoints for each partner instance that was trained in stage 1
        # Additionally, create new fcp agent for each team
        # Apart from FCP agent, all other networks will be frozen
        partners = []
        partner_states = []
        for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(teams):
            team_partners = []
            team_partner_states = []

            if cls_team_fcp_agents[team_ix]:
                rng, _rng = jax.random.split(rng)
                fcp_agent, fcp_agent_state = cls_team_fcp_agents[team_ix](
                    _rng, config,
                    env_spec, teams[team_ix], env,
                    f"fcp-{team_ix}_"
                )
                team_partners.append(fcp_agent)
                team_partner_states.append(fcp_agent_state)

            for p_ix in range(partner_count):
                for ckpt_step in checkpoint_load_steps:
                    rng, _rng = jax.random.split(rng)
                    checkpoint_prefix = f"{team_ix}-{p_ix}_"
                    partner, init_partner_state = cls_SelfPlayAgent(
                        _rng, config,
                        env_spec, teams[team_ix], env,
                        checkpoint_prefix
                        )
                    loaded_partner_state, fn_frozen_update, fn_frozen_save = partner.load(init_partner_state, ckpt_step)
                    frozen_partner = SelfPlayAgent(
                        partner.get_action,
                        fn_frozen_update,
                        fn_frozen_save,
                        partner.load
                    )
                    team_partners.append(frozen_partner)
                    team_partner_states.append(loaded_partner_state)

            partners.append(team_partners)
            partner_states.append(team_partner_states)

        # There is no need to update the TeamSpec for each of the teams
        # since this variable won't be used on that way again


        # Randomly sample partners to assign to agents in their respective teams for each environment
        # Generate this using numpy so that jax can treat it as static values
        # Important for later splitting up the arrays
        # ====
        # Key difference to Stage 1 mapping is that all 'partners' are bumped by one index value (+1)
        # Each environment will only have fcp agent (the one to be trained) 
        # This will be sampled randomly (which requires a large number of environments to make good training progress)

        # Randomly select one agent that will be the fcp target agent
        # But only if that fcp agent is defined for the given team
        target_agents = [ env.agents[_i] for _i in np_rng.choice(env.num_agents, env_spec.count) ]

        map_agent_uid_to_partner_instance = dict()
        for (team_ix, team_partners) in enumerate(partners):
            for agent_id in teams[team_ix].agent_uids:
                partner_count = len(team_partners)
                env_instance_count = env_spec.count

                # If we want to train an fcp agent for this team
                if cls_team_fcp_agents[team_ix]:
                    # Add +1 after sampling because we want to ignore the target fcp agent at index 0
                    # Choice range is -1 actual to account for that
                    sampled_partners = np_rng.choice(partner_count-1, size=(env_instance_count, ))+1
                    # Replace the 'sampled_partners' with 0 index according to 'fcp_agent_map'
                    for _i in range(sampled_partners.shape[0]):
                        if target_agents[_i] == agent_id:
                            sampled_partners[_i] = 0
                    # Continue as with stage 1 with masking
                    masks = []
                    for i in range(partner_count):
                        masks.append(np.isclose(sampled_partners, i))
                    map_agent_uid_to_partner_instance[agent_id] = ( team_ix, masks, partner_count )
                else:
                    # Otherwise use the same sampling as in Stage 1
                    sampled_partners = np_rng.choice(partner_count, size=(env_instance_count, ))
                    masks = []
                    for i in range(partner_count):
                        masks.append(np.isclose(sampled_partners, i))
                    map_agent_uid_to_partner_instance[agent_id] = ( team_ix, masks, partner_count )

        for agent_id, entry in map_agent_uid_to_partner_instance.items():
            for mask in entry[1]:
                assert np.any(mask), f"Error Agent mask with no assigned environments, Either increase the number of environments or try a different seed."

         # Scan over episodes
        episode_runner_state = env_obsv_state, partner_states, rng
        last_episode_runner_state, (episode_metrics, ) = jax.lax.scan(
            _make_episode(config, env_spec, teams, env, partners, map_agent_uid_to_partner_instance),
            episode_runner_state,
            jnp.arange(0, config["NUM_EPISODES"]), # Used to keep track of saved network parameters
            length=config["NUM_EPISODES"]
        )

        unravelled_episode_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(jnp.concatenate([ x[i] for i in range(x.shape[0]) ]), axis=-1), episode_metrics)
        
        return unravelled_episode_metrics, partners

    return _stage_2



class FCP:
    @staticmethod
    def make_stage_1(config, env_spec: EnvSpec, teams: list[TeamSpec], numpy_seed:int):
        _make_stage_1(config, env_spec, teams, numpy_seed)(jax.random.PRNGKey(0))
        raise ValueError("Oops forgot to re-jit")
        return jax.jit(_make_stage_1(config, env_spec, teams, numpy_seed))

    @staticmethod
    def make_stage_2(
        config,
        env_spec: EnvSpec, teams: list[TeamSpec],
        cls_team_fcp_agents: list[None|Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any, Any, Any], 'SelfPlayAgent']],
        checkpoint_load_steps: list[int],
        numpy_seed: int
        ):
        return jax.jit(_make_stage_2(config, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps, numpy_seed))