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
        x = self.max_action * nn.tanh(x)
        # Add some noise and return a multivariate normal distribution that can be sampled from
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return distrax.MultivariateNormalDiag(x, jnp.exp(log_std))
    
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
    
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    # info: jnp.ndarray # TODO Figure out if we need this
    
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

    tx_actor = optax.adam(config["LR"], eps=1e-5)
    tx_critic = optax.adam(config["LR"], eps=1e-5)

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
    def _delayed_policy_update(update_state, batch_traj):
        rng, train_states, batch_count = update_state
        state_actor, state_actor_target, state_critic, state_critic_target = train_states
        done, action, reward, obs, next_obs = batch_traj

        # Compute actor loss
        # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        def _loss_actor(actor_params, actor_apply_fn, rng, state_critic, obs):
            # TODO verify that this is actually correct
            # it might be that both actor AND critic losses are updated here? (but that can't really be)
            rng, _rng = jax.random.split(rng)
            act = actor_apply_fn(actor_params, obs).sample(seed=_rng)
            obs_action = jnp.concatenate([obs, act], axis=1)
            critic_value_Q1 = state_critic.apply_fn(state_critic.params, obs_action)[0]
            return -jnp.mean(critic_value_Q1)
        
        actor_grad_fn = jax.value_and_grad(_loss_actor)
        rng, _rng = jax.random.split(rng)
        actor_loss, actor_grads = actor_grad_fn(state_actor.params, state_actor.apply_fn, _rng, state_critic, obs)
        state_actor = state_actor.apply_gradients(grads=actor_grads)

        # Update the frozen target models
        state_critic_target = state_critic_target.replace(params=jax.tree_map(
            lambda param, target_param: config["TAU"]*param + (1-config["TAU"])*target_param,
            state_critic.params, state_critic_target.params
        ))
        state_actor_target = state_actor_target.replace(params=jax.tree_map(
            lambda param, target_param: config["TAU"]*param + (1-config["TAU"])*target_param,
            state_actor.params, state_actor_target.params
        ))

        train_states = (state_actor, state_actor_target, state_critic, state_critic_target)
        update_state = rng, train_states, batch_count
        return update_state, actor_loss
    return _delayed_policy_update

def _make_actor_critic_policy_update(config):
    def _policy_update(update_state, batch_traj):
        pass
    return _policy_update


def _make_td3_update_batch(config):
    def _update_batch(update_state, batch_traj):
        rng, train_states, batch_count = update_state
        state_actor, state_actor_target, state_critic, state_critic_target = train_states
        done, action, reward, obs, next_obs = batch_traj

        # Select action according to policy (actor target) (noise is already included??)
        rng, _rng = jax.random.split(rng)
        next_pi = state_actor_target.apply_fn(state_actor_target.params, next_obs)
        next_action = next_pi.sample(seed=_rng)

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

        # Perform delayed policy updates
        train_states = (state_actor, state_actor_target, state_critic, state_critic_target)
        update_state = rng, train_states, batch_count

        update_state, actor_loss = jax.lax.cond(
            batch_count % config["POLICY_FREQ"] == 0,
            _make_delayed_policy_update(config),
            lambda updt_state, traj: (updt_state, jnp.nan),
            update_state, batch_traj
        )
        

        rng, train_states, batch_count = update_state
        state_actor, state_actor_target, state_critic, state_critic_target = train_states

        loss_info = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss
        }

        train_states = (state_actor, state_actor_target, state_critic, state_critic_target)
        batch_count = batch_count + 1
        update_state = rng, train_states, batch_count
        return update_state, loss_info

    return _update_batch

def make_td3_update(config):
    def _td3_update(rng: jax.dtypes.prng_key, train_states, traj_buffer):
        # Split the replay buffer into shuffled batches
        flattened_traj_buffer = jax.tree_map(
            lambda x: x.reshape((config["NUM_BATCHES"]*config["BATCH_SIZE"], ) + x.shape[2:]),
            traj_buffer
        )
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, config["NUM_BATCHES"]*config["BATCH_SIZE"])
        shuffled_flattened_traj_buffer = jax.tree_map(
            lambda x: jnp.take(x, permutation, axis=0),
            flattened_traj_buffer
        )
        batches = jax.tree_map(
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

# def make_actor_critic(
#     rng,
#     cls_actor: Type[nn.Module], args_actor: list[Any], kwargs_actor: dict[str, Any],
#     init_x_actor,
#     cls_critic: Type[nn.Module], args_critic: list[Any], kwargs_critic: dict[str, Any],
#     init_x_critic,
#     learn_rate=3e-4
#     ):
#     actor = cls_actor(*args_actor, **kwargs_actor)
#     rng, _rng = jax.random.split(rng)
#     actor_params = actor.init(_rng, init_x_actor)
#     actor_init_state = TrainState.create(
#         apply_fn=actor.apply,
#         params=actor_params,
#         tx=optax.adam(learn_rate, eps=1e-5)
#     )

#     critic = cls_critic(*args_critic, **kwargs_critic)
#     rng, _rng = jax.random.split(rng)
#     critic_params = critic.init(_rng, init_x_critic)
#     critic_init_state = TrainState.create(
#         apply_fn=critic.apply,
#         params=critic_params,
#         tx=optax.adam(learn_rate, eps=1e-5)
#     )

#     return actor_init_state, critic_init_state



# def make_td3(
#     state_dim: int, action_dim: int, actor_module: nn.Module, critic_module: nn.Module,
#     discount=0.99,
#     tau=0.005,
#     policy_noise=0.2,
#     noise_clip=0.5,
#     policy_freq=2
#     ):
#     rng = jax.random.PRNGKey(0)
#     rng, _rng = jax.random.split(rng)
#     actor_state, critic_state = make_actor_critic(_rng, None, None, None, None, None, None, None, None)
#     actor_target_state = actor_state
#     critic_target_state = critic_state

#     def train(batch_size):
#         # Sample replay buffer
#         done, actions, aux_data, reward, observations, info = trajectories

#         # Select action according to policy and add clipped noise
#         rng, _rng = jax.random.split(rng)
#         # Original implementation adds noise to get more "random" actions
#         # However jax allows you to simply "sample" from the action space which should give it a more random distribution
#         # noise = jnp.clip(( jax.random.normal(_rng, actions.shape) * policy_noise ), -noise_clip, noise_clip)
#         # noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)	
# 		# next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

