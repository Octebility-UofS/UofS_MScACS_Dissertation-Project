from typing import Any, NamedTuple, Sequence, Type
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import numpy as np

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
    
class TrainStates(NamedTuple):
    state_actor: TrainState
    state_actor_target: TrainState
    state_critic: TrainState
    state_critic_target: TrainState
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    # info: jnp.ndarray # TODO Figure out if we need this

def batchify(x: dict, agent_list: list[str], num_actors: int):
    x = jnp.stack([ x[a] for a in agent_list ])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list: list[str], num_envs: int, num_actors: int):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
    
def make_td3(rng: jax.dtypes.prng_key, config, env):
    net_actor = DefaultActor(env.action_space(env.agents[0]).shape[0], config["MAX_ACTION"])
    net_critic = DefaultCritic()

    net_actor_init_x = jnp.zeros(env.observation_space(env.agents[0]).shape[-1])
    net_critic_init_x = jnp.concatenate([
        jnp.zeros(env.observation_space(env.agents[0]).shape[-1]),
        jnp.zeros(env.action_space(env.agents[0]).shape[0])
    ])

    rng, _rng = jax.random.split(rng)
    net_actor_params = net_actor.init(_rng, net_actor_init_x)
    rng, _rng = jax.random.split(rng)
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
    return (
        train_state_actor,
        train_state_actor_target,
        train_state_critic,
        train_state_critic_target
    )


def _make_delayed_policy_update(config):
    def _delayed_policy_update(train_states, batch_traj):
        state_actor, state_actor_target, state_critic, state_critic_target = train_states
        done, action, reward, obs, next_obs = batch_traj

        # Compute actor loss
        # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        def _loss_actor(actor_params, actor_apply_fn, state_critic, obs):
            # TODO verify that this is actually correct
            # it might be that both actor AND critic losses are updated here? (but that can't really be)
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
        return train_states, actor_loss
    return _delayed_policy_update


def _make_td3_update_batch(config):
    def _update_batch(rng, train_states, batch_traj, counter):
        state_actor, state_actor_target, state_critic, state_critic_target = train_states
        done, action, reward, obs, next_obs = batch_traj

        # # Select action according to policy and add clipped noise
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
        target_Q = reward[:, jnp.newaxis] + (1-done[:, jnp.newaxis]) * config["DISCOUNT"] * target_Q

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

        # Perform delayed policy updates
        def _no_policy_update(train_states, traj):
            return train_states, jnp.nan

        train_states, actor_loss = jax.lax.cond(
            (counter % config["POLICY_FREQ"]) == 0,
            _make_delayed_policy_update(config),
            _no_policy_update,
            train_states, batch_traj
        )
        
        loss_info = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss
        }

        return train_states, loss_info

    return _update_batch

def make_td3_update(config):
    # def _pure_random()

    def _sample_trajectories(rng, trajectories):
        sample_ixs = jax.random.randint(rng, (config["BATCH_SIZE"], ), 0, trajectories.done.shape[0])
        samples = jax.tree.map(
            lambda x: x[sample_ixs],
            trajectories
        )
        return jax.tree.map(
            lambda x: x.reshape( (x.shape[0]*x.shape[1], ) + x.shape[2:] ),
            samples
        )
    
    def _td3_update(rng: jax.dtypes.prng_key, train_states, trajectories, counter):
        # Get random samples from the trajectories
        rng, _rng = jax.random.split(rng)
        # samples = _sample_trajectories(_rng, trajectories)
        samples = jax.tree.map(
            lambda x: x.reshape( (x.shape[0]*x.shape[1], ) + x.shape[2:] ),
            trajectories
        )

        # Perform the update using the samples
        rng, _rng = jax.random.split(rng)
        train_states, loss_info = _make_td3_update_batch(
            config
        )(_rng, train_states, samples, counter)

        return train_states, loss_info


        # Split the replay buffer into shuffled batches
        flattened_traj_buffer = jax.tree.map(
            lambda x: x.reshape((config["NUM_BATCHES"]*config["BATCH_SIZE"], ) + x.shape[2:]),
            traj_buffer
        )
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, config["NUM_BATCHES"]*config["BATCH_SIZE"])
        shuffled_flattened_traj_buffer = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=0),
            flattened_traj_buffer
        )
        batches = jax.tree.map(
            lambda x: x.reshape( (config["NUM_BATCHES"], config["BATCH_SIZE"]) + x.shape[1:] ),
            shuffled_flattened_traj_buffer
        )
        
        rng, _rng = jax.random.split(rng)
        batch_count = 0
        update_state = _rng, train_states, batch_count
        # Perform batch updates
        update_state, loss_info = jax.lax.scan(
            _make_td3_update_batch(config),
            update_state, batches
        )

        _rng, train_states, batch_count = update_state
        return train_states, loss_info
    return _td3_update