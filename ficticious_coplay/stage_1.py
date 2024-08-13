import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

from ficticious_coplay.common import EnvSpec, TeamSpec, _make_episode, transform_env_partner, untransform_env_partner
from util.util import rec_frozenset


def _generate_mappings(config, env_spec: EnvSpec, teams: list[TeamSpec]):
    np_rng = np.random.default_rng(config["NUMPY_SEED"])

    # Split the environment instances up into evenly sized ranges
    # We are implementing 'true' self play so in each team, each policy is only trained against itself
    env_instance_count = env_spec.count
    map_agent_uid_to_partner_instance = dict()
    for (team_ix, team_spec) in enumerate(teams):
        map_agent_uid_to_partner_instance[team_ix] = dict()
        partner_count = team_spec.agent_count
        # get evenly spaced 'slicing indexes'
        # according to number of partners and environments
        slice_ixs = np.linspace(0, env_instance_count, partner_count+1, dtype=int)
        for agent_id in team_spec.agent_uids:
            for p_ix in range(partner_count):
                if p_ix not in map_agent_uid_to_partner_instance[team_ix]:
                    map_agent_uid_to_partner_instance[team_ix][p_ix] = dict()
                map_agent_uid_to_partner_instance[team_ix][p_ix][agent_id] = (slice_ixs[p_ix], slice_ixs[p_ix+1])
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


def make_stage_1(config, init_rng: jax.dtypes.prng_key, env_spec: EnvSpec, teams: list[TeamSpec]):

    env = jaxmarl.make(env_spec.env_id, **env_spec.env_kwargs)

    # Create all parallelised partner agents for each team
    partners = []
    partner_states = []
    for team_ix, (cls_SelfPlayAgent, partner_count, _) in enumerate(teams):
        team_partners = []
        team_partner_states = []
        for p_ix in range(partner_count):
            init_rng, _rng = jax.random.split(init_rng)
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

    (
        map_agent_uid_to_partner_instance,
        reverse_map_agent_uid_to_partner_instance
    ) = _generate_mappings(config, env_spec, teams)

    def _stage_1(rng):
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

    return _stage_1