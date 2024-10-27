from functools import partial
from typing import NamedTuple, Sequence
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import optax

class TrainStates(NamedTuple):
    state_actor: TrainState
    state_actor_target: TrainState
    state_critic: TrainState
    state_critic_target: TrainState
    
class Trajectory(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray




class DefaultActor(nn.Module):
    action_dim: Sequence[int]
    max_action: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return self.max_action * nn.tanh(x)
    
class DefaultCritic(nn.Module):
    @nn.compact
    def __call__(self, obs_action):
        x = obs_action

        # For Q1
        q1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        q1 = nn.relu(q1)
        q1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q1)

        # For Q2
        q2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        q2 = nn.relu(q2)
        q2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q2)
        return q1, q2




class TD3:
    @staticmethod
    def init(rng_init, config, env, cls_actor=DefaultActor, cls_critic=DefaultCritic):
        net_actor = DefaultActor(env.action_space(env.agents[0]).shape[0], config["MAX_ACTION"])
        net_critic = DefaultCritic()

        net_actor_init_x = jnp.zeros(env.observation_space(env.agents[0]).shape[-1])
        net_critic_init_x = jnp.concatenate([
            jnp.zeros(env.observation_space(env.agents[0]).shape[-1]),
            jnp.zeros(env.action_space(env.agents[0]).shape[0])
        ])

        rng_init, _rng = jax.random.split(rng_init)
        net_actor_params = net_actor.init(_rng, net_actor_init_x)
        rng_init, _rng = jax.random.split(rng_init)
        net_critic_params = net_critic.init(_rng, net_critic_init_x)

        tx_actor = optax.adam(config["LR"], eps=1e-8)
        tx_critic = optax.adam(config["LR"], eps=1e-8)

        train_state_actor = TrainState.create(
            apply_fn=net_actor.apply,
            params=net_actor_params,
            tx=tx_actor
        )

        train_state_critic = TrainState.create(
            apply_fn=net_critic.apply,
            params=net_critic_params,
            tx=tx_critic
        )

        train_state_actor_target = train_state_actor
        train_state_critic_target = train_state_critic
        return TrainStates(
            train_state_actor,
            train_state_actor_target,
            train_state_critic,
            train_state_critic_target
        )
    
    @staticmethod
    def make_update(config):
        def _update(rng, train_states: TrainStates, trajectories: Trajectory, delayed_policy_update: bool):
            state_actor, state_actor_target, state_critic, state_critic_target = train_states
            done, action, reward, obs, next_obs = trajectories

            # Select action according to policy and add clipped noise
            rng, _rng = jax.random.split(rng)
            noise = jnp.clip(
                jax.random.normal(_rng, action.shape) * config["POLICY_NOISE"],
                min=-config["NOISE_CLIP"], max=config["NOISE_CLIP"]
            )
            next_action = jnp.clip(
                noise + state_actor_target.apply_fn(state_actor_target.params, next_obs),
                min=-config["MAX_ACTION"], max=config["MAX_ACTION"]
            )

            # Compute the target Q value
            next_obs_action = jnp.concatenate([next_obs, next_action], axis=1)
            target_Q1, target_Q2 = state_critic_target.apply_fn(state_critic_target.params, next_obs_action)
            target_Q = jnp.minimum(target_Q1, target_Q2)
            target_Q = reward + (1-done) * config["DISCOUNT"] * target_Q

            # Loss Function for updating Critic
            def _loss_critic(critic_params, critic_apply_fn, obs, action, target_Q):
                # Get current Q estimates
                obs_action = jnp.concatenate([obs, action], axis=1)
                current_Q1, current_Q2 =  critic_apply_fn(critic_params, obs_action)
                # Loss Function is mean Squared error (mean of L2 loss)
                loss = jnp.mean(optax.l2_loss(current_Q1, target_Q)) + jnp.mean(optax.l2_loss(current_Q2, target_Q))
                return loss
            
            # Compute critic loss and apply gradients / optimize
            critic_grad_fn = jax.value_and_grad(_loss_critic)
            critic_loss, critic_grads = critic_grad_fn(state_critic.params, state_critic.apply_fn, obs, action, target_Q)
            state_critic = state_critic.apply_gradients(grads=critic_grads)

            train_states = TrainStates(
                state_actor,
                state_actor_target,
                state_critic,
                state_critic_target
            )

            func_no_policy_update = lambda t_states, trajs: t_states
            train_states = jax.lax.cond(
                delayed_policy_update,
                TD3._make_delayed_policy_update(config),
                func_no_policy_update,
                train_states, trajectories
            )
            
            return train_states
        return _update
    
    @staticmethod
    def _make_delayed_policy_update(config):
        def _delayed_update(train_states: TrainStates, trajectories: Trajectory):
            state_actor, state_actor_target, state_critic, state_critic_target = train_states
            done, action, reward, obs, next_obs = trajectories

            # Compute actor loss
            # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            def _loss_actor(actor_params, actor_apply_fn, state_critic, obs):
                act = actor_apply_fn(actor_params, obs)
                obs_action = jnp.concatenate([obs, act], axis=1)
                critic_value_Q1 = state_critic.apply_fn(state_critic.params, obs_action)[0]
                return -jnp.mean(critic_value_Q1)
            
            actor_grad_fn = jax.value_and_grad(_loss_actor)
            actor_loss, actor_grads = actor_grad_fn(state_actor.params, state_actor.apply_fn, state_critic, obs)
            state_actor = state_actor.apply_gradients(grads=actor_grads)

            # Update the frozen target models
            state_critic_target = state_critic_target.replace(params=jax.tree.map(
                lambda param, target_param: config["TAU"]*param + (1-config["TAU"])*target_param,
                state_critic.params, state_critic_target.params
            ))
            state_actor_target = state_actor_target.replace(params=jax.tree.map(
                lambda param, target_param: config["TAU"]*param + (1-config["TAU"])*target_param,
                state_actor.params, state_actor_target.params
            ))

            train_states = TrainStates(
                state_actor,
                state_actor_target,
                state_critic,
                state_critic_target
            )
            return train_states
        return _delayed_update
    



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
    @partial(jax.jit, static_argnames=['batch_size',])
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




def batchify(x: dict, agent_list: list[str], num_actors: int):
    x = jnp.stack([ x[a] for a in agent_list ])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list: list[str], num_envs: int, num_actors: int):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_env_step_exploration(config, env):
    def _env_step(runner_state, _):
        rng, last_env_obs, last_env_state = runner_state

        # Sample random actions
        rng, _rng = jax.random.split(rng)
        rng_actions = jax.random.split(_rng, (env.num_agents, config["NUM_ENVS"]))
        env_action = {
            agent: jax.vmap(env.action_space(agent).sample, in_axes=0)(rng_actions[i])
            for i, agent in enumerate(env.agents)
        }

        # Step environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        env_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, last_env_state, env_action)

        trajectories = Trajectory(
            batchify(done, env.agents, config["NUM_ACTORS"]),
            batchify(env_action, env.agents, config["NUM_ACTORS"]),
            batchify(reward, env.agents, config["NUM_ACTORS"]),
            batchify(last_env_obs, env.agents, config["NUM_ACTORS"]),
            batchify(env_obs, env.agents, config["NUM_ACTORS"]),
        )

        runner_state = rng, env_obs, env_state
        return runner_state, trajectories
    return _env_step

def make_env_step_train(config, env):
    def _env_step(rng, state_actor, last_env_obs, last_env_state):
        last_env_obs_batch = batchify(last_env_obs, env.agents, config["NUM_ACTORS"])

        # Select Deterministic action
        action_batch = state_actor.apply_fn(state_actor.params, last_env_obs_batch)
        # Add exploration noise and clip to maximum action value
        rng, _rng = jax.random.split(rng)
        action_batch = jnp.clip(
            action_batch + ( config["EXPLORATION_NOISE"] * jax.random.normal(_rng, action_batch.shape) ),
            min=-config["MAX_ACTION"], max=config["MAX_ACTION"]
        )
        # Transform back to 'env space'
        env_action = unbatchify(action_batch, env.agents, config["NUM_ENVS"], env.num_agents)

        # Step environment
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        env_obs, env_state, reward, done, info = jax.vmap(env.step)(
            rng_step, last_env_state, env_action
        )

        trajectories = Trajectory(
            batchify(done, env.agents, config["NUM_ACTORS"]),
            action_batch,
            batchify(reward, env.agents, config["NUM_ACTORS"]),
            last_env_obs_batch,
            batchify(env_obs, env.agents, config["NUM_ACTORS"]),
        )
        return env_obs, env_state, trajectories
    return _env_step
    
def make_iteration(config, env):
    def _train_step(runner_state, bool_delayed_policy_update):
        rng, train_states, replay_buffer, env_obs, env_state = runner_state

        # Simulate environments and add to replay buffer
        rng, _rng = jax.random.split(rng)
        env_obs, env_state, trajectories = make_env_step_train(config, env)(
            _rng, train_states.state_actor, env_obs, env_state
        )
        replay_buffer = ReplayBufferTD3.add(replay_buffer, trajectories)

        # Sample Trajectories from replay buffer and perform update
        rng, _rng = jax.random.split(rng)
        rng, rng_sample = jax.random.split(rng)
        buffer_samples = ReplayBufferTD3.sample(rng_sample, replay_buffer, config["BATCH_SIZE"])
        train_states = TD3.make_update(config)(
            _rng, train_states,
            buffer_samples,
            bool_delayed_policy_update
        )

        metric = {} # TODO

        runner_state = rng, train_states, replay_buffer, env_obs, env_state
        return runner_state, metric
    return _train_step

def make_train(config, rng_init):
    # Create Environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env, replace_info=True)

    # Compute Other Configurations
    assert(
        config["TOTAL_STEPS"] >= config["EXPLORATION_STEPS"]
    ), "The number of exploration steps must not exceed the total number of timesteps"
    assert(
        ( config["TOTAL_STEPS"] % config["NUM_ENVS"] ) == 0
    ), "The Total Number of Timesteps must be divisble by the Number of Environments"
    assert(
        ( config["EXPLORATION_STEPS"] % config["NUM_ENVS"] ) == 0
    ), "The Number of Exploration Steps must be divisble by the Number of Environments"
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_ITERATIONS_EXPLORATION"] = int(config["EXPLORATION_STEPS"]) // int(config["NUM_ENVS"])
    _remaining_steps = int(config["TOTAL_STEPS"]) - int(config["EXPLORATION_STEPS"])
    config["NUM_ITERATIONS_TRAIN"] = _remaining_steps // int(config["NUM_ENVS"])

    # Set up checkpointing
    if config["NUM_CHECKPOINTS"]:
        config["_CHECKPOINT_STEPS"] = list(np.linspace(
            0,
            config["NUM_ITERATIONS_TRAIN"] - 1,
            num=config["NUM_CHECKPOINTS"],
            endpoint=True,
            dtype=np.int32
        ))
    else:
        config["_CHECKPOINT_STEPS"] = []

    # Init TD3 & Network
    # We are assuming that the continuous action space is symmetric
    config["MAX_ACTION"] = env.action_spaces[env.agents[0]].high
    train_states = TD3.init(rng_init, config, env)

    for k, v in config.items():
        print(k, v)
    print("\n")
    
    def _train(rng, train_states: TrainStates):
        # Init Environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_obs, env_state = jax.vmap(env.reset)(reset_rng)

        # Perform random exploration, init Replay Buffer, and add trajectories to replay buffer
        rng, _rng = jax.random.split(rng)
        runner_state = _rng, env_obs, env_state
        runner_state, exploration_trajectories = jax.lax.scan(
            make_env_step_exploration(config, env),
            runner_state, None,
            length=config["NUM_ITERATIONS_EXPLORATION"]
        )
        exploration_trajectories_unbatched = jax.tree.map(
            lambda x: jnp.concatenate(x, axis=0),
            exploration_trajectories
        )
        replay_buffer = ReplayBufferTD3.init(
            config["TOTAL_STEPS"] * config["NUM_ACTORS"],
            batched_entry=exploration_trajectories_unbatched
        )
        replay_buffer = ReplayBufferTD3.add(replay_buffer, exploration_trajectories_unbatched)
        env_obs, env_state = runner_state[1], runner_state[2]

        # Perform training iterations
        # During each iteration, one env simulation across all environments is done,
        # and trajectories added to replay buffer
        # Then perform TD3 updates according to config
        rng, _rng = jax.random.split(rng)
        bool_a_delayed_policy_updates = jnp.array(
            [i%config["POLICY_FREQ"] for i in range(config["NUM_ITERATIONS_TRAIN"])],
            dtype=jnp.bool
        )
        runner_state = _rng, train_states, replay_buffer, env_obs, env_state
        runner_state, metrics = jax.lax.scan(
            make_iteration(config, env),
            runner_state, bool_a_delayed_policy_updates,
            length=config["NUM_ITERATIONS_TRAIN"]
        )
        train_states, replay_buffer, env_obs, env_state = runner_state[1:]

        metrics = {
            'train_reward': replay_buffer['data'].reward
        }
        return metrics
    return train_states, _train


@hydra.main(version_base=None, config_path="config", config_name="td3_ff_mabrax")
def main(config):
    config = OmegaConf.to_container(config)

    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(config["DISABLE_JIT"]):
        rng, _rng = jax.random.split(rng)
        train_states, train_fn = make_train(config, _rng)
        train_jit = jax.jit(train_fn)
        metrics = train_jit(rng, train_states)

        plt.plot(range(metrics['train_reward'].shape[0]), metrics['train_reward'])
        plt.show()


if __name__ == "__main__":
    main()