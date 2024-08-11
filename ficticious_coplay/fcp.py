from typing import Callable, Optional

import jax
import jax.experimental
import jaxmarl
import numpy as np

from ficticious_coplay.common import (EnvSpec, SelfPlayAgentFactory, TeamSpec,
                                      transform_env_partner,
                                      untransform_env_partner)
from ficticious_coplay.stage_1 import make_stage_1
from ficticious_coplay.stage_2 import make_stage_2
from util.util import rec_frozenset


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












class FCP:
    @staticmethod
    def make_stage_1(config, env_spec: EnvSpec, teams: list[TeamSpec]):
        return jax.jit(make_stage_1(config, env_spec, teams))

    @staticmethod
    def make_stage_2(
        config, env_spec: EnvSpec, teams: list[TeamSpec],
        cls_team_fcp_agents: list[Optional[SelfPlayAgentFactory]],
        checkpoint_load_steps: list[int]
        ):
        return jax.jit(make_stage_2(config, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps))
