from functools import partial
from typing import Any, Callable, NamedTuple, Union

import jax
import jax.experimental
import jax.numpy as jnp
import jaxmarl
import numpy as np

from ficticious_coplay.common import EnvSpec, SelfPlayAgent, TeamSpec, _make_episode, transform_env_partner, untransform_env_partner
from ficticious_coplay.util.util import rec_frozenset





def get_rollout(config, rng, env_spec: EnvSpec, team_agents: list[list[str]], team_assignments: list[list[tuple[tuple[str, int], Callable]]], max_steps=200):
    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)
    max_steps = min(max_steps, env.max_steps)

    np_rng = np.random.default_rng(0)

    partners = []
    partner_states = []
    for team_ix, assignment in enumerate(team_assignments):
        partners.append([])
        partner_states.append([])
        for ((p_prefix, p_load_step), p_load_cls) in assignment:
            rng, _rng = jax.random.split(rng)
            partner, init_partner_state = p_load_cls(
                _rng, config, env_spec, None, env, # set team_spec[team_ix] to None for now since it doesn't really seem to be used
                p_prefix
            )
            loaded_partner_state, _, _ = partner.load(init_partner_state, p_load_step)
            partners[team_ix].append(partner)
            partner_states[team_ix].append(loaded_partner_state)

    env_instance_count = env_spec.count
    map_agent_uid_to_partner_instance = dict()
    for (team_ix, agents) in enumerate(team_agents):
        map_agent_uid_to_partner_instance[team_ix] = dict()
        for p_ix, agent_id in enumerate(agents):
            map_agent_uid_to_partner_instance[team_ix][p_ix] = {agent_id: (0, env_instance_count)}
        # Convert dict of agent id's to slices into a list sorted by agent id
        for partner_ix, agent_mapping in map_agent_uid_to_partner_instance[team_ix].items():
            map_agent_uid_to_partner_instance[team_ix][partner_ix] = [
                item for item in sorted(agent_mapping.items(), key=lambda t: t[0])
            ]

    reverse_map_agent_uid_to_partner_instance = dict()
    for (team_ix, agents) in enumerate(team_agents):
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

    @jax.jit
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
    rng_reset = jax.random.split(_rng, env_spec.count)
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
    state_seq = []
    reward_seq = []
    for step_ix in range(max_steps):
        state_seq.append(jax.tree.map(
            lambda x: x[step_ix],
            env_states
        ))
        reward_seq.append(jax.tree.map(
            lambda x: x[step_ix],
            rewards
        ))

    return env_states, rewards




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

        return episode_metrics, last_episode_runner_state

    return _stage_1



def _make_stage_2(
    config,
    env_spec: EnvSpec, team_specs: list[TeamSpec],
    cls_team_fcp_agents: list[Union[None,Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any, Any, Any], SelfPlayAgent]]],
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
        
        return episode_metrics, partners

    return _stage_2



class FCP:
    @staticmethod
    def make_stage_1(config, env_spec: EnvSpec, teams: list[TeamSpec], numpy_seed:int):
        return _make_stage_1(config, env_spec, teams, numpy_seed)

    @staticmethod
    def make_stage_2(
        config,
        env_spec: EnvSpec, teams: list[TeamSpec],
        cls_team_fcp_agents: list[Union[None,Callable[[Any, Any, list[EnvSpec], 'TeamSpec', Any, Any, Any], 'SelfPlayAgent']]],
        checkpoint_load_steps: list[int],
        numpy_seed: int
        ):
        return _make_stage_2(config, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps, numpy_seed)
