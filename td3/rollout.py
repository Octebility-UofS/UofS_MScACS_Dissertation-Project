import jax
import jax.numpy as jnp
from brax.io import html

from td3.td3 import batchify, unbatchify

def _make_env_step(config, env):
    def _env_step(runner_state, _):
        state_actor, env_state, last_obs, rng = runner_state

        obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

        # Select action
        rng, _rng = jax.random.split(rng)
        pi = state_actor.apply_fn(state_actor.params, obs_batch)
        action = pi.sample(seed=_rng)
        # log_prob = pi.log_prob(action) # Probably not needed, maybe only for IPPO
        env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        
        # Step environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            rng_step, env_state, env_act
        )

        runner_state = (state_actor, env_state, obsv, rng)
        return runner_state, (env_state, reward)

    return _env_step

def make_rollout(config, env):

    def _rollout(rng, actor_state):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_state, env_state, obsv, rng)
        runner_state, (rollout_states, rollout_rewards) = jax.lax.scan(
            _make_env_step(config, env),
            runner_state, None,
            length=config["NUM_STEPS"]
        )

        # Select rollout with highest reward
        ix = jnp.argmax(jnp.sum(rollout_rewards['__all__'], axis=0))
        init_state = jax.tree.map(lambda x: x[ix], env_state)
        select_states = jax.tree.map(lambda x: x[:, ix], rollout_states)

        concat_states = jax.tree.map(
            lambda x, y: jnp.concatenate([x.reshape((1, ) + x.shape), y], axis=0),
            init_state, select_states
        )
        return concat_states
    return _rollout

def visualise(env, states, num_steps, out_path):
    state_history = [
        jax.tree.map(lambda x: x[i], states.pipeline_state) for i in range(num_steps+1)
    ]
    rendered_html = html.render(env.sys, state_history)
    with open(out_path, 'w') as f:
        f.write(rendered_html)