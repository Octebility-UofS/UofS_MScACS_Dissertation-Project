
from functools import partial
import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

class ReplayBufferTD3:
    """
    JAX implementation of a 'growing' replay buffer.
    The buffer must be initialised with a maximum size 
    and a sample structure of the pytree that makes up the replay buffer.
    Entries can be added as singles or as batches.
    The replay buffer can be sampled randomly with replacement,
    the sampling respects the 'current size' of the replay buffer.
    """

    @staticmethod
    def init(max_size: int, single_entry=None, batched_entry=None):
        if single_entry is None and batched_entry is None:
            raise ValueError("You must specify one of 'single_entry' or 'batched_entry'")
        elif not (single_entry is None or batched_entry is None):
            raise ValueError("You must not only specify one of 'single_entry' or 'batched_entry'")
        
        data = None
        if single_entry is not None:
            data = jax.tree.map(
                lambda arr: jnp.zeros_like(arr, shape=(int(max_size), ) + arr.shape),
                single_entry
            )
        else:
            data = jax.tree.map(
                lambda arr: jnp.zeros_like(arr, shape=(int(max_size), ) + arr.shape[1:]),
                batched_entry
            )

        return {
            'max_size': jnp.array(max_size),
            'current_size': jnp.array(0),
            'data': data
        }
    
    @staticmethod
    def add(buffer, tree):
        tree_size = jax.tree.flatten(tree)[0][0].shape[0]
        updated_data = jax.tree.map(
            lambda arr, tree_arr: jax.lax.dynamic_update_slice_in_dim(
                arr, tree_arr,
                start_index=buffer['current_size'], axis=0
            ),
            buffer['data'], tree
        )
        return {
            'max_size': buffer['max_size'],
            'current_size': buffer['current_size'] + tree_size,
            'data': updated_data
        }

    @staticmethod
    @partial(jax.jit, static_argnames=['batch_size', 'num_minibatches'])
    def sample(rng, buffer, batch_size):
        # Return a single batch of size 'batch_size'
        # Since we want to sample only from the buffer elements that are actually populated,
        # we can't just sample from the entire buffer
        # Instead, we need to use a pure callback to sample indexes within the populated range
        # And then dynamically select these indexes
        choice_indexes = jax.pure_callback(
            lambda _rng, data_size: jax.random.choice(
                _rng, data_size.item(),
                shape=(batch_size, ), replace=False
            ),
            jax.ShapeDtypeStruct((batch_size, ), jnp.int32),
            rng, buffer['current_size']
        )
        return jax.tree.map(
            lambda x: jax.vmap(partial(jax.lax.dynamic_index_in_dim, x, keepdims=False))(choice_indexes),
            buffer['data']
        )
    

config = {
    "ENV_NAME": "ant_4x2",
    "ENV_KWARGS": {},
    "NUM_ENVS": 128,
}


def main():
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env, replace_info=True)

    def _actual_main(rng):
        dummy_data = {'agent_0': jnp.array([1, 1]), 'agent_1': jnp.array([1, 1]), 'agent_2': jnp.array([1, 1]), 'agent_3': jnp.array([1, 1])}
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, statev = jax.vmap(env.reset)(reset_rng)

        replay_buffer = ReplayBufferTD3.init(512, batched_entry=obsv)

        # Sample random actions
        rng, _rng = jax.random.split(rng)
        action_rng = jax.random.split(_rng, (env.num_agents, config["NUM_ENVS"]))
        actions = {agent: jax.vmap(env.action_space(agent).sample)(action_rng[i]) for i, agent in enumerate(env.agents)}

        rng, _rng = jax.random.split(rng)
        step_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, statev, rewardv, donev, infov = jax.vmap(env.step)(step_rng, statev, actions)
        dummy_data = jax.tree.map(lambda x: x+1, dummy_data)
        replay_buffer = ReplayBufferTD3.add(replay_buffer, obsv)

        rng, _rng = jax.random.split(rng)
        step_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, statev, rewardv, donev, infov = jax.vmap(env.step)(step_rng, statev, actions)
        dummy_data = jax.tree.map(lambda x: x+1, dummy_data)
        replay_buffer = ReplayBufferTD3.add(replay_buffer, obsv)

        # dummy_data = jax.tree.map(lambda x: x+1, dummy_data)
        # replay_buffer = ReplayBufferTD3.add(replay_buffer, dummy_data)

        # dummy_data = jax.tree.map(lambda x: x+1, dummy_data)
        # replay_buffer = ReplayBufferTD3.add(replay_buffer, dummy_data)

        rng, _rng = jax.random.split(rng)
        print(ReplayBufferTD3.sample(_rng, replay_buffer, 64))

        rng, _rng = jax.random.split(rng)
        ReplayBufferTD3.sample(_rng, replay_buffer, 64, num_minibatches=4)

        return None
    return _actual_main



if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    res = jax.jit(main())(rng)
    print(res)