import jax
import jax.numpy as jnp
import jaxmarl
from ficticious_coplay.common import EnvSpec, TeamSpec
from jaxmarl.environments.overcooked import overcooked_layouts

def make_runner_rollout(config, env, partners, map_agent_id_to_partner, num_total_envs, num_permutations):
    @jax.jit
    def _runner_rollout(runner_state, _):
        obsv, statev, partner_states, rng = runner_state

        # Reshape observations to be read by each partner properly
        obs_batch = jax.tree.map(
            lambda x: x.reshape((num_permutations, config["ROLLOUT_NUM_ENVS"]) + x.shape[1:]),
            obsv
        )

        permutation_actions = []
        updated_partner_states = []
        for permutation_ix, permutation_partners in enumerate(partners):
            permutation_actions.append({})
            updated_partner_states.append(
                [ [None]*len(team_partners) for team_partners in permutation_partners ]
            )
            for agent_id, (team_ix, p_ix) in map_agent_id_to_partner.items():
                rng, _rng = jax.random.split(rng)
                agent_observations = obs_batch[agent_id][permutation_ix]
                agent_state, agent_actions, agent_aux_adata = partners[permutation_ix][team_ix][p_ix].get_action(
                    _rng,
                    agent_observations,
                    agent_observations.reshape( (agent_observations.shape[0], -1) ), # flattened observations
                    partner_states[permutation_ix][team_ix][p_ix]
                )
                permutation_actions[permutation_ix][agent_id] = agent_actions
                updated_partner_states[permutation_ix][team_ix][p_ix] = agent_state

        # 'Glue' the batches back together
        unbatched_actions = jax.tree.map(
            lambda *xs: jnp.concatenate(xs),
            *permutation_actions
        )
        
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_total_envs)
        updated_obsv, updated_statev, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, statev, unbatched_actions)

        runner_state = updated_obsv, updated_statev, updated_partner_states, rng

        batch_fn = lambda x: x.reshape((num_permutations, config["ROLLOUT_NUM_ENVS"]) + x.shape[1:])
        batched_statev = jax.tree.map(batch_fn, updated_statev)
        batched_reward = jax.tree.map(batch_fn, reward)
        return runner_state, (batched_statev, batched_reward)


    return _runner_rollout


def make_rollout(config, init_rng, rollout_env_spec, team_agents, rollout_permutations, max_steps=300):
    
    env = jaxmarl.make(rollout_env_spec.env_id, **rollout_env_spec.env_kwargs)

    rollout_permutation_partners = []
    rollout_permutation_partner_states = []
    for rollout_ix, rollout_permutation in enumerate(rollout_permutations):
        rollout_permutation_partners.append([])
        rollout_permutation_partner_states.append([])
        for team_ix, team_assignments in enumerate(rollout_permutation):
            rollout_permutation_partners[rollout_ix].append([])
            rollout_permutation_partner_states[rollout_ix].append([])
            for (p_prefix, p_load_step), p_load_cls in team_assignments:
                init_rng, _rng = jax.random.split(init_rng)
                partner, init_partner_state = p_load_cls(
                    _rng, config, rollout_env_spec, None, env, # set team_spec[team_ix] to None for now since it doesn't really seem to be used
                    p_prefix
                )
                loaded_partner_state, _, _ = partner.load(init_partner_state, p_load_step)
                rollout_permutation_partners[rollout_ix][team_ix].append(partner)
                rollout_permutation_partner_states[rollout_ix][team_ix].append(loaded_partner_state)

    map_agent_id_to_partner = {}
    for team_ix, agent_ids in enumerate(team_agents):
        for p_ix, a_id in enumerate(agent_ids):
            map_agent_id_to_partner[a_id] = (team_ix, p_ix)

    total_environments = config["ROLLOUT_NUM_ENVS"] * len(rollout_permutations)

    @jax.jit
    def _rollout(rng):
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, total_environments)
        obsv, statev = jax.vmap(env.reset, in_axes=(0, ))(rng_reset)

        runner_state = obsv, statev, rollout_permutation_partner_states, rng
        last_runner_state, (scanned_states, scanned_rewards) = jax.lax.scan(
            make_runner_rollout(config, env, rollout_permutation_partners, map_agent_id_to_partner, total_environments, len(rollout_permutations)),
            runner_state, None,
            length=max_steps
        )
        batched_init_state = jax.tree.map(
            lambda x: x.reshape((len(rollout_permutations), config["ROLLOUT_NUM_ENVS"]) + x.shape[1:]),
            statev
            )
        combined_states = jax.tree.map(
            lambda x, y: jnp.concatenate([x[jnp.newaxis, :], y]),
            batched_init_state, scanned_states
        )
        return combined_states, scanned_rewards

    return _rollout