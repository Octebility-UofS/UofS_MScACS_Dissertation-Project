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
    map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[int, tuple[tuple[str, tuple[int, int]], ...]]]]],
    reverse_map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[str, tuple[tuple[tuple[int, int], tuple[int, tuple[int, int]]], ...]]]]],
    env_spec: EnvSpec, env,
    partners: list[list[SelfPlayAgent]]
    ):
    @jax.jit
    def _envs_step(runner_state, _):
        env_obsv_state, partner_states, rng = runner_state

        # Transform all observation vectors to group them by partner agents
        env_obsv, env_state = env_obsv_state
        transformed_observations = transform_env_partner(env_obsv, map_agent_uid_to_partner_instance)


        # Run each transformed stack of observations through their respective parntner agents
        updated_partner_states = []
        tree_partner_actions = dict()
        tree_partner_aux_data = dict()
        for team_ix, team_partners in enumerate(partners):
            team_updated_partner_states = []
            tree_partner_actions[team_ix] = dict()
            tree_partner_aux_data[team_ix] = dict()
            for p_ix, partner in enumerate(team_partners):
                agent_state = partner_states[team_ix][p_ix]
                partner_observations = transformed_observations[team_ix][p_ix]
                flattened_partner_observations = partner_observations.reshape((partner_observations.shape[0], -1))

                rng, _rng = jax.random.split(rng)
                agent_state, agent_actions, agent_aux_data = partner.get_action(
                    _rng,
                    partner_observations,
                    flattened_partner_observations,
                    agent_state
                    )

                team_updated_partner_states.append(agent_state)
                tree_partner_actions[team_ix][p_ix] = agent_actions
                tree_partner_aux_data[team_ix][p_ix] = agent_aux_data
            updated_partner_states.append(team_updated_partner_states)


        # Reverse the process with the stacked actions to use them for stepping the environments
        env_act = untransform_env_partner(tree_partner_actions, reverse_map_agent_uid_to_partner_instance)
        
        #STEP ENVIRONMENTS
        env_obsv, env_state = env_obsv_state
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, env_spec.count)
        new_env_obsv, new_env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0,0,0)
            )(rng_step, env_state, env_act)
        updated_env_obsv_state = new_env_obsv, new_env_state

        # COLLECT TRANSITIONS FOR EACH AGENT
        # Will be later passed to update functions for each agent
        transformed_reward = transform_env_partner(reward, map_agent_uid_to_partner_instance)
        transformed_done = transform_env_partner(done, map_agent_uid_to_partner_instance)

        agent_transitions = []
        for team_ix in range(len(partners)):
            agent_transitions.append([])
            for p_ix in range(len(partners[team_ix])):
                agent_transitions[team_ix].append(Transition(
                    transformed_done[team_ix][p_ix],
                    tree_partner_actions[team_ix][p_ix],
                    tree_partner_aux_data[team_ix][p_ix],
                    transformed_reward[team_ix][p_ix],
                    transformed_observations[team_ix][p_ix],
                    {} # TODO what to do with 'info'? info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                ))

        
        runner_state = updated_env_obsv_state, updated_partner_states, rng
        return runner_state, agent_transitions


    return _envs_step


def _make_update_step(
    config,
    map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[int, tuple[tuple[str, tuple[int, int]], ...]]]]],
    reverse_map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[str, tuple[tuple[tuple[int, int], tuple[int, tuple[int, int]]], ...]]]]],
    env_spec: EnvSpec, teams: list[TeamSpec], env,
    partners: list[list[SelfPlayAgent]]
    ):
    @jax.jit
    def _update_step(runner_state: tuple[Any, Any, list[SelfPlayAgent], list[int], Any], save_counter):
        # Collect Trajectories
        runner_state, trajectories = jax.lax.scan(
            _make_envs_step(
                map_agent_uid_to_partner_instance,
                reverse_map_agent_uid_to_partner_instance,
                env_spec, env, partners),
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


def _make_episode(
    config, env_spec: EnvSpec, teams: list[TeamSpec], env, partners,
    map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[int, tuple[tuple[str, tuple[int, int]], ...]]]]],
    reverse_map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[str, tuple[tuple[tuple[int, int], tuple[int, tuple[int, int]]], ...]]]]]
    ):
    def _episode(runner_episode, counter):
        env_obsv_state, partner_states, rng = runner_episode

        rng, _rng = jax.random.split(rng)

        runner_state = env_obsv_state, partner_states, _rng
        last_runner_state, (metrics, ) = jax.lax.scan(
                _make_update_step(
                    config,
                    map_agent_uid_to_partner_instance,
                    reverse_map_agent_uid_to_partner_instance,
                    env_spec, teams, env, partners
                    ),
                runner_state,
                counter*config["NUM_UPDATES"] + jnp.arange(0, config["NUM_UPDATES"]), # Keep track of individual update step counter
                length=config["NUM_UPDATES"]
            )
        env_obsv_state, partner_states, rng = last_runner_state

        runner_episode = env_obsv_state, partner_states, rng
        return runner_episode, (metrics, )
    
    return _episode


def get_rollout(config, env_spec: EnvSpec, team_specs: list[TeamSpec], team_fcp_agents, rollout_load_config, max_steps=200):
    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)
    max_steps = min(max_steps, env.max_steps)

    np_rng = np.random.default_rng(0)
    rng = jax.random.PRNGKey(0)

    teams = [
        TeamSpec(
            spec.agent_class,
            len(spec.agent_uids),
            spec.agent_uids
            ) for spec in team_specs
    ]

    partners = []
    partner_states = []
    for team_ix, team_config in enumerate(rollout_load_config):
        team_partners = []
        team_partner_states = []
        for partner_prefix, checkpoint_step in team_config:
            rng, _rng = jax.random.split(rng)
            load_cls = team_fcp_agents[team_ix] if partner_prefix.startswith("fcp-") else team_specs[team_ix].agent_class
            partner, init_partner_state = load_cls(
                _rng, config, env_spec, teams[team_ix], env,
                partner_prefix
                )
            loaded_partner_state, _, _ = partner.load(init_partner_state, checkpoint_step)
            team_partner_states.append(loaded_partner_state)
            team_partners.append(partner)
        partners.append(team_partners)
        partner_states.append(team_partner_states)


    env_instance_count = 1
    map_agent_uid_to_partner_instance = dict()
    for (team_ix, team_spec) in enumerate(teams):
        map_agent_uid_to_partner_instance[team_ix] = dict()
        partner_count = team_spec.agent_count
        for p_ix, agent_id in enumerate(team_spec.agent_uids):
            map_agent_uid_to_partner_instance[team_ix][p_ix] = {agent_id: (0, env_instance_count)}
        # Convert dict of agent id's to slices into a list sorted by agent id
        for partner_ix, agent_mapping in map_agent_uid_to_partner_instance[team_ix].items():
            map_agent_uid_to_partner_instance[team_ix][partner_ix] = [
                item for item in sorted(agent_mapping.items(), key=lambda t: t[0])
            ]

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

    # The maps need to be hashable in order to be used as static arguments by partial jax.jit
    # So we are transform dictionaries and lists into immutable datatypes (frozenset and tuple)
    map_agent_uid_to_partner_instance = rec_frozenset(map_agent_uid_to_partner_instance)
    reverse_map_agent_uid_to_partner_instance = rec_frozenset(reverse_map_agent_uid_to_partner_instance)

    def _runner_rollout(runner_state, _):
        env_obsv_state, partner_states, rng = runner_state

        # Transform all observation vectors to group them by partner agents
        env_obsv, env_state = env_obsv_state
        transformed_observations = transform_env_partner(env_obsv, map_agent_uid_to_partner_instance)

        # Run each transformed stack of observations through their respective parntner agents
        updated_partner_states = []
        tree_partner_actions = dict()
        tree_partner_aux_data = dict()
        for team_ix, team_partners in enumerate(partners):
            team_updated_partner_states = []
            tree_partner_actions[team_ix] = dict()
            tree_partner_aux_data[team_ix] = dict()
            for p_ix, partner in enumerate(team_partners):
                agent_state = partner_states[team_ix][p_ix]
                partner_observations = transformed_observations[team_ix][p_ix]
                flattened_partner_observations = partner_observations.reshape((partner_observations.shape[0], -1))

                rng, _rng = jax.random.split(rng)
                agent_state, agent_actions, agent_aux_data = partner.get_action(
                    _rng,
                    partner_observations,
                    flattened_partner_observations,
                    agent_state
                    )

                team_updated_partner_states.append(agent_state)
                tree_partner_actions[team_ix][p_ix] = agent_actions
                tree_partner_aux_data[team_ix][p_ix] = agent_aux_data
            updated_partner_states.append(team_updated_partner_states)


        # Reverse the process with the stacked actions to use them for stepping the environments
        env_act = untransform_env_partner(tree_partner_actions, reverse_map_agent_uid_to_partner_instance)
        
        #STEP ENVIRONMENTS
        env_obsv, env_state = env_obsv_state
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, env_spec.count)
        new_env_obsv, new_env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0,0,0)
            )(rng_step, env_state, env_act)
        updated_env_obsv_state = new_env_obsv, new_env_state

        # Collect new runner state and record state
        runner_state = updated_env_obsv_state, updated_partner_states, rng
        return runner_state, (new_env_state, reward)


    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, 1)
    obs, state = jax.vmap(env.reset, in_axes=(0, ))(rng_reset)

    state_seq = [state, ]
    reward_seq = [{}, ]
    runner_state = (obs, state), partner_states, rng
    _, (env_states, rewards) = jax.lax.scan(
        _runner_rollout,
        runner_state, None,
        length=max_steps
    )
    # Unbatchify the states since they're currently of shape [1, :]
    state_seq += [ jax.tree_map(lambda x: x[i], env_states) for i in range(max_steps) ]
    reward_seq += [ jax.tree_map(lambda x: x[i], rewards) for i in range(max_steps) ]
    unbatched_state_seq = [ jax.tree_map(lambda x: x[0], state) for state in state_seq ]
    unbatched_reward_seq = [ jax.tree_map(lambda x: x[0], reward) for reward in reward_seq ]

    return unbatched_state_seq, unbatched_reward_seq




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
        env_instance_count = env_spec.count
        map_agent_uid_to_partner_instance = dict()
        for (team_ix, team_spec) in enumerate(teams):
            map_agent_uid_to_partner_instance[team_ix] = dict()
            partner_count = team_spec.agent_count
            # get evenly spaced 'slicing indexes'
            # according to number of partners and environments
            slice_ixs = np.linspace(0, env_instance_count, partner_count+1, dtype=int)
            for agent_id in team_spec.agent_uids:
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

        # The maps need to be hashable in order to be used as static arguments by partial jax.jit
        # So we are transform dictionaries and lists into immutable datatypes (frozenset and tuple)
        map_agent_uid_to_partner_instance = rec_frozenset(map_agent_uid_to_partner_instance)
        reverse_map_agent_uid_to_partner_instance = rec_frozenset(reverse_map_agent_uid_to_partner_instance)

        # Scan over episodes
        episode_runner_state = env_obsv_state, partner_states, rng
        last_episode_runner_state, (episode_metrics, ) = jax.lax.scan(
            _make_episode(
                config, env_spec, teams, env, partners,
                map_agent_uid_to_partner_instance,
                reverse_map_agent_uid_to_partner_instance
                ),
            episode_runner_state,
            jnp.arange(0, config["NUM_EPISODES"]), # Used to keep track of saved network parameters
            length=config["NUM_EPISODES"]
        )

        unravelled_episode_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(jnp.concatenate([ x[i] for i in range(x.shape[0]) ]), axis=-1), episode_metrics)

        return unravelled_episode_metrics, last_episode_runner_state

    return _stage_1



def _make_stage_2(
    config,
    env_spec: EnvSpec, team_specs: list[TeamSpec],
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

        # Additionally, create new fcp agent for each team
        # Apart from FCP agent, all other networks will be frozen
        partners = []
        partner_states = []
        for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(team_specs):
            team_partners = []
            team_partner_states = []

            if cls_team_fcp_agents[team_ix]:
                rng, _rng = jax.random.split(rng)
                fcp_agent, fcp_agent_state = cls_team_fcp_agents[team_ix](
                    _rng, config,
                    env_spec, team_specs[team_ix], env,
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
                        env_spec, team_specs[team_ix], env,
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

        # Update the TeamSpec for each team to properly reflect the number of instances
        # We simply update the agent_count in the TeamSpec
        teams = [
            TeamSpec(
                team_specs[team_ix].agent_class,
                len(team_partners),
                team_specs[team_ix].agent_uids
                ) for team_ix, team_partners in enumerate(partners)
        ]

        # There is no need to update the TeamSpec for each of the teams
        # since this variable won't be used on that way again


        # Split the environment instances up into evenly sized ranges
        # For each agent in each team, ranges are randomly assigned for each partner instance
        # This vastly simplifies the complexity of transforming observations
        # While still keeping an element of randomness
        env_instance_count = env_spec.count
        # ====
        # Assign slices to each fcp partner so that they are assigned to all agents within their team
        num_fcp_assignments = sum(
            [ len(teams[team_ix].agent_uids) for team_ix, mk_fn in enumerate(cls_team_fcp_agents) if mk_fn ]
            )
        raw_fcp_slice_ixs = np.linspace(0, env_instance_count, num_fcp_assignments+1, dtype=int)
        fcp_slice_ixs = dict()
        _counter = 0
        for team_ix, mk_fn in enumerate(cls_team_fcp_agents):
            if mk_fn:
                fcp_slice_ixs[team_ix] = dict()
                for agent_id in teams[team_ix].agent_uids:
                    fcp_slice_ixs[team_ix][agent_id] = (raw_fcp_slice_ixs[_counter], raw_fcp_slice_ixs[_counter+1])
                    _counter += 1
            else:
                fcp_slice_ixs[team_ix] = None
        # ====
        map_agent_uid_to_partner_instance = dict()
        for (team_ix, team_spec) in enumerate(teams):
            map_agent_uid_to_partner_instance[team_ix] = dict()
            partner_count = team_spec.agent_count

            # We need to act differently depending on whether an FCP partner is present
            if fcp_slice_ixs[team_ix]:
                fcp_partner_ix = 0
                # Assign FCP partner a range for each agent in the team
                for agent_id in team_spec.agent_uids:
                    if fcp_partner_ix not in map_agent_uid_to_partner_instance[team_ix]:
                        map_agent_uid_to_partner_instance[team_ix][fcp_partner_ix] = dict()
                    map_agent_uid_to_partner_instance[team_ix][fcp_partner_ix][agent_id] = fcp_slice_ixs[team_ix][agent_id]
                # Assign the remaining partners
                for agent_id in team_spec.agent_uids:
                    # Create slice indexes for the other partners with the remaining ranges
                    slice_ixs = None
                    # First check if fcp range is at lower bound or upper bound
                    if fcp_slice_ixs[team_ix][agent_id][0] == 0:
                        # In this case, we can use a simple linspace
                        # Use partner_count instead of partner_count+1 since we don't need to assign to FCP agent
                        _linspace_slices = np.linspace(fcp_slice_ixs[team_ix][agent_id][1], env_instance_count, partner_count, dtype=int)
                        slice_ixs = [ (s0, s1) for (s0, s1) in zip(_linspace_slices, _linspace_slices[1:]) ]
                    elif fcp_slice_ixs[team_ix][agent_id][1] == env_instance_count:
                        # In this case, we can use a simple linspace
                        # Use partner_count instead of partner_count+1 since we don't need to assign to FCP agent
                        _linspace_slices = np.linspace(0, fcp_slice_ixs[team_ix][agent_id][0], partner_count, dtype=int)
                        slice_ixs = [ _slice for _slice in zip(_linspace_slices, _linspace_slices[1:]) ]
                    else:
                        # We need to splice together two linspace arrays
                        # First calculate ratio between the two ranges and assign linspace accordingly
                        lower_range = (0, fcp_slice_ixs[team_ix][agent_id][0])
                        upper_range = (fcp_slice_ixs[team_ix][agent_id][1], env_instance_count)
                        lower_range_ratio = ( (lower_range[1]-lower_range[0]) + (upper_range[1]-upper_range[0]) ) / (lower_range[1]-lower_range[0])
                        total_range_partners = partner_count - 1 # Since we take away one FCP partner
                        lower_range_partners = total_range_partners // lower_range_ratio
                        upper_range_partners = total_range_partners - lower_range_partners
                        lower_linspace = np.linspace(lower_range[0], lower_range[1], lower_range_partners+1, dtype=int)
                        upper_linspace = np.linspace(upper_range[0], upper_range[1], upper_range_partners+1, dtype=int)
                        slice_ixs = (
                            [ _slice for _slice in zip(lower_linspace, lower_linspace[1:]) ]
                            + [ _slice for _slice in zip(upper_linspace, upper_linspace[1:]) ]
                            )
                    # Now that we have the slices, create a permutation of the available partners and assignt the slices
                    # skip index 0 since that's the fcp partner
                    partner_instance_permutation = np_rng.permutation(np.arange(1, partner_count))
                    for ix, partner_ix in enumerate(partner_instance_permutation):
                        if partner_ix not in map_agent_uid_to_partner_instance[team_ix]:
                            map_agent_uid_to_partner_instance[team_ix][partner_ix] = dict()
                        map_agent_uid_to_partner_instance[team_ix][partner_ix][agent_id] = slice_ixs[ix]
            else:
                # Otherwise, we don't have an fcp partner and we can generate the index as normal
                _linspace_slices = np.linspace(0, env_instance_count, partner_count+1, dtype=int)
                slice_ixs = [ (s0, s1) for (s0, s1) in zip(_linspace_slices, _linspace_slices[1:]) ]
                for agent_id in team_spec.agent_uids:
                    # get a random permutation of assigning partner instances to slices
                    partner_instance_permutation = np_rng.permutation(np.arange(partner_count))
                    for ix, partner_ix in enumerate(partner_instance_permutation):
                        if partner_ix not in map_agent_uid_to_partner_instance[team_ix]:
                            map_agent_uid_to_partner_instance[team_ix][partner_ix] = dict()
                        map_agent_uid_to_partner_instance[team_ix][partner_ix][agent_id] = slice_ixs[ix]
            # Convert dict of agent id's to slices into a list sorted by agent id
            for partner_ix, agent_mapping in map_agent_uid_to_partner_instance[team_ix].items():
                map_agent_uid_to_partner_instance[team_ix][partner_ix] = [
                    item for item in sorted(agent_mapping.items(), key=lambda t: t[0])
                ]

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

        # The maps need to be hashable in order to be used as static arguments by partial jax.jit
        # So we are transform dictionaries and lists into immutable datatypes (frozenset and tuple)
        map_agent_uid_to_partner_instance = rec_frozenset(map_agent_uid_to_partner_instance)
        reverse_map_agent_uid_to_partner_instance = rec_frozenset(reverse_map_agent_uid_to_partner_instance)


         # Scan over episodes
        episode_runner_state = env_obsv_state, partner_states, rng
        last_episode_runner_state, (episode_metrics, ) = jax.lax.scan(
            _make_episode(
                config, env_spec, teams, env, partners,
                map_agent_uid_to_partner_instance,
                reverse_map_agent_uid_to_partner_instance
                ),
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
        return _make_stage_1(config, env_spec, teams, numpy_seed)
        return jax.jit(_make_stage_1(config, env_spec, teams, numpy_seed))

    @staticmethod
    def make_stage_2(
        config,
        env_spec: EnvSpec, teams: list[TeamSpec],
        cls_team_fcp_agents: list[None|Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any, Any, Any], 'SelfPlayAgent']],
        checkpoint_load_steps: list[int],
        numpy_seed: int
        ):
        return _make_stage_2(config, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps, numpy_seed)
        return jax.jit(_make_stage_2(config, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps, numpy_seed))
