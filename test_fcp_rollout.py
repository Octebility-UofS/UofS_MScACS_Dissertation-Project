import sys
import os

if __name__ == "__main__":
    if sys.argv[-1] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
       

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from omegaconf import OmegaConf
import yaml
import jaxmarl

from ficticious_coplay.common import EnvSpec, TeamSpec
from test_fcp import make_ppo_agent

def parseScientificNumber(obj):
    if type(obj) == list:
        return [ parseScientificNumber(l) for l in obj ]
    elif type(obj) == dict:
        return { k: parseScientificNumber(v) for k, v in obj.items() }
    elif type(obj) == str:
        splt = obj.split("e")
        if len(splt) == 2:
            try:
                return (float(obj))
            except ValueError:
                pass
        if obj.isdigit():
            return int(obj)
        return obj
    else:
        return obj
        

def load_partner(params_path, rng, config, rollout_env_spec, env):
    fragments = os.path.split(params_path)[1].split("_")
    p_prefix = "_".join(fragments[:-1]) + "_"
    load_step = fragments[-1].split(".")[0]
    name = p_prefix+load_step

    partner, init_partner_state = make_ppo_agent(
        rng, config, rollout_env_spec, None, env, # set team_spec[team_ix] to None for now since it doesn't really seem to be used
        p_prefix
    )
    loaded_partner_state, _, _ = partner.load(init_partner_state, load_step, dir=os.path.split(params_path)[0])
    return partner, loaded_partner_state, name

def make_runner_rollout(config, env, partners, map_agent_id_to_partner):
    @jax.jit
    def _runner_rollout(runner_state, _):
        obsv, statev, partner_states, rng = runner_state

        actions = {}
        updated_partner_states = [[]]
        for agent_ix, agent_id in enumerate(env.agents):
            rng, _rng = jax.random.split(rng)
            agent_state, agent_actions, agent_aux_data = partners[0][agent_ix].get_action(
                _rng,
                obsv[agent_id],
                obsv[agent_id].reshape( (obsv[agent_id].shape[0], -1) ), # flattened observations
                partner_states[0][agent_ix]
            )
            actions[agent_id] = agent_actions
            updated_partner_states[0].append(agent_state)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["ROLLOUT_NUM_ENVS"])
        updated_obsv, updated_statev, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, statev, actions)

        runner_state = updated_obsv, updated_statev, updated_partner_states, rng
        return runner_state, (updated_statev, reward)
    return _runner_rollout

def make_rollout(config, env, partners, partner_states, map_agent_id_to_partner):
    def _rollout(rng):
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, config["ROLLOUT_NUM_ENVS"])
        obsv, statev = jax.vmap(env.reset, in_axes=(0, ))(rng_reset)

        runner_state = obsv, statev, partner_states, rng
        last_runner_state, (scanned_states, scanned_rewards) = jax.lax.scan(
            make_runner_rollout(config, env, partners, map_agent_id_to_partner),
            runner_state, None,
            length=300
        )
        select_ix = jnp.argmax(jnp.sum(scanned_rewards[env.agents[0]], axis=0))
        init_state = jax.tree.map(lambda x: x[select_ix:select_ix+1], statev)
        selected_states = jax.tree.map(lambda x: x[:, select_ix], scanned_states)
        return jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            init_state, selected_states
        )
    return _rollout

def main():

    params_0 = sys.argv[1]
    params_1 = sys.argv[2]
    out_path = sys.argv[3]
    
    rng = jax.random.PRNGKey(0)
    
    config = None
    with open(os.path.join("config", "test_fcp.yaml"), 'r') as f:
        config = OmegaConf.to_container(
            OmegaConf.create(
                parseScientificNumber(yaml.load(f.read(), yaml.BaseLoader))
            )
        )        

    config["ROLLOUT_NUM_ENVS"] = 25
    
    rollout_env_spec = EnvSpec(
        config["ENV"]["ID"],
        config["ROLLOUT_NUM_ENVS"],
        {"layout": overcooked_layouts[config["ENV"]["KWARGS"]["layout"]]}
    )

    teams = []
    for _team_ix, team_config in config["TEAMS"].items():
        teams.append(TeamSpec(
            make_ppo_agent,
            team_config["AGENT_COUNT"],
            team_config["AGENT_IDS"]
        ))

    env = jaxmarl.make(rollout_env_spec.env_id, **rollout_env_spec.env_kwargs)
    
    rng, _rng = jax.random.split(rng)
    p0, p0_state, p0_name = load_partner(params_0, _rng, config, rollout_env_spec, env)
    rng, _rng = jax.random.split(rng)
    p1, p1_state, p1_name = load_partner(params_1, _rng, config, rollout_env_spec, env)

    partners = [[
        p0, p1
    ]]
    partner_states = [[
        p0_state, p1_state
    ]]

    map_agent_id_to_partner = {}
    for ix, agent_id in enumerate(env.agents):
        map_agent_id_to_partner[agent_id] = (0, ix)

    rng, _rng = jax.random.split(rng)
    states = make_rollout(config, env, partners, partner_states, map_agent_id_to_partner)(_rng)

    steps = 300
    state_seq = [ jax.tree.map(lambda x: x[i], states) for i in range(steps+1) ]
    viz =  OvercookedVisualizer()
    viz.animate(state_seq, env.agent_view_size, filename=os.path.join(out_path, f"animation.{p0_name}.{p1_name}.gif"))

if __name__ == "__main__":
    main()