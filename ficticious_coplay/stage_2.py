from typing import Any, Callable, Optional, Union
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from ficticious_coplay.common import EnvSpec, SelfPlayAgent, SelfPlayAgentFactory, TeamSpec, _make_episode, transform_env_partner, untransform_env_partner
from util.util import rec_frozenset

def _generate_mappings(config, env_spec: EnvSpec, teams: list[TeamSpec], cls_team_fcp_agents: list[Optional[SelfPlayAgentFactory]]):
    np_rng = np.random.default_rng(config["NUMPY_SEED"])

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

    return (
        map_agent_uid_to_partner_instance,
        reverse_map_agent_uid_to_partner_instance
    )


    

def make_stage_2(
    config, init_rng: jax.dtypes.prng_key,
    env_spec: EnvSpec, team_specs: list[TeamSpec],
    cls_team_fcp_agents: list[Optional[SelfPlayAgentFactory]],
    checkpoint_load_steps: list[int]
    ):

    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)

    # Additionally, create new fcp agent for each team
    # Apart from FCP agent, all other networks will be frozen
    partners = []
    partner_states = []
    for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(team_specs):
        team_partners = []
        team_partner_states = []

        if cls_team_fcp_agents[team_ix]:
            init_rng, _rng = jax.random.split(init_rng)
            fcp_agent, fcp_agent_state = cls_team_fcp_agents[team_ix](
                _rng, config,
                env_spec, team_specs[team_ix], env,
                f"fcp-{team_ix}_"
            )
            team_partners.append(fcp_agent)
            team_partner_states.append(fcp_agent_state)

        for p_ix in range(partner_count):
            for ckpt_step in checkpoint_load_steps:
                init_rng, _rng = jax.random.split(init_rng)
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

    teams = []
    for team_ix, (cls_SelfPlayAgent, partner_count, agent_uids) in enumerate(team_specs):
        teams.append(TeamSpec(
            cls_SelfPlayAgent,
            partner_count*len(checkpoint_load_steps) + (1 if cls_team_fcp_agents[team_ix] else 0),
            agent_uids
        ))

    (
        map_agent_uid_to_partner_instance,
        reverse_map_agent_uid_to_partner_instance
    ) = _generate_mappings(config, env_spec, teams, cls_team_fcp_agents)

    def _stage_2(rng):
       # Create all parallelised environments
        rng_array = jax.random.split(rng, env_spec.count)
        env_obsv_state = jax.vmap(env.reset, in_axes=(0, ))(rng_array)        


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

    return _stage_2