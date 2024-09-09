from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, NewType, Union

import jax
import jax.numpy as jnp
from omegaconf import DictKeyType, OmegaConf

class EnvSpec(NamedTuple):
    env_id: str
    count: int
    env_kwargs: Any

JaxmarlEnv = NewType("JaxmarlEnv", Any)

def SelfPlayAgentFactory(init_rng: jax.dtypes.prng_key, config: dict[DictKeyType, Any], env_spec: 'EnvSpec', team_spec: 'TeamSpec', env: Any, checkpoint_prefix: str) -> 'SelfPlayAgent':
    return NotImplementedError("This is merely a type function")

class TeamSpec(NamedTuple):
    agent_class: SelfPlayAgentFactory
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
                    {} # TODO what to do with 'info'? info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                ))

        
        runner_state = updated_env_obsv_state, updated_partner_states, rng
        return runner_state, agent_transitions


    return _envs_step


def _make_update_step(
    config,
    map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[int, tuple[tuple[str, tuple[int, int]], ...]]]]],
    reverse_map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[str, tuple[tuple[tuple[int, int], tuple[int, tuple[int, int]]], ...]]]]],
    env_spec: EnvSpec, teams: list[TeamSpec], env,
    partners: list[list[SelfPlayAgent]],
    reward_milestone_map: dict[str, int] # Allows us to count the occurence of specific 'milestones' that are attached to a unique reward value
    ):
    @jax.jit
    def _update_step(runner_state: tuple[Any, Any, list[SelfPlayAgent], list[int], Any], save_counter):        
        metrics = {}

        # Collect Trajectories
        with jax.disable_jit():
            runner_state, trajectories = jax.lax.scan(
                _make_envs_step(
                    map_agent_uid_to_partner_instance,
                    reverse_map_agent_uid_to_partner_instance,
                    env_spec, env, partners),
                runner_state, None,
                length=config["ENV_STEPS"]
            )

        # Log the reward for each agent policy
        metrics["reward"] = {
            "sum": {},
            "dishes": {}
        }
        for team_ix, team_partners in enumerate(partners):
            metrics["reward"]["sum"][team_ix] = {}
            for milestone, value in reward_milestone_map.items():
                metrics["reward"][milestone][team_ix] = {}
            
            for p_ix, partner in enumerate(team_partners):
                # Sum the reward for all environment steps in this episode in order to save some memory
                metrics["reward"]["sum"][team_ix][p_ix] = jnp.sum(trajectories[team_ix][p_ix].reward, axis=0)
                for milestone, value in reward_milestone_map.items():
                    metrics["reward"][milestone][team_ix][p_ix] = jnp.sum(trajectories[team_ix][p_ix].reward == value, axis=0)


        # Update each agent given their trajectories
        metrics["update_metrics"] = {}
        env_obsv_state, partner_states, rng = runner_state
        updated_partner_states = []
        for team_ix, team_partners in enumerate(partners):
            metrics["update_metrics"][team_ix] = {}
            team_updated_partner_states = []
            for p_ix, partner in enumerate(team_partners):
                rng, _rng = jax.random.split(rng)
                updated_partner_state, update_metrics = partner.update(_rng, trajectories[team_ix][p_ix], partner_states[team_ix][p_ix])
                metrics["update_metrics"][team_ix][p_ix] = update_metrics
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
    reverse_map_agent_uid_to_partner_instance: frozenset[tuple[int, frozenset[tuple[str, tuple[tuple[tuple[int, int], tuple[int, tuple[int, int]]], ...]]]]],
    reward_milestone_map: dict[str, int] # Allows us to count the occurence of specific 'milestones' that are attached to a unique reward value
    ):
    @jax.jit
    def _episode(runner_episode, counter):
        env_obsv_state, partner_states, rng = runner_episode

        rng, _rng = jax.random.split(rng)

        runner_state = env_obsv_state, partner_states, _rng
        last_runner_state, (metrics, ) = jax.lax.scan(
                _make_update_step(
                    config,
                    map_agent_uid_to_partner_instance,
                    reverse_map_agent_uid_to_partner_instance,
                    env_spec, teams, env, partners,
                    reward_milestone_map
                    ),
                runner_state,
                counter*config["NUM_UPDATES"] + jnp.arange(0, config["NUM_UPDATES"]), # Keep track of individual update step counter
                length=config["NUM_UPDATES"]
            )
        env_obsv_state, partner_states, rng = last_runner_state

        runner_episode = env_obsv_state, partner_states, rng
        return runner_episode, (metrics, )
    
    return _episode