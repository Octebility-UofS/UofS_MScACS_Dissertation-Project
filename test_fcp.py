from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import os
import pickle
import sys
from typing import Any, NamedTuple, Sequence, Type

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import optax
import orbax.checkpoint
from flax.training import orbax_utils
import matplotlib.pyplot as plt

from ficticious_coplay.fcp import FCP, EnvMapping, EnvSpec, SelfPlayAgent, AgentUID, TeamSpec, _make_stage_2, get_rollout


class SimpleNetwork(nn.Module):
    output_dim: Sequence[int] # action_space_dim
    activation = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            raise ValueError(f"Activation function for '{self.activation}' is not defined")
        x = nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)

        pi_logits = x
        pi_logits = nn.Dense(self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(pi_logits)
        pi = distrax.Categorical(logits=pi_logits)

        value = x
        value = nn.Dense(8, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value = activation(value)
        value = nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)
        value_squeezed = jnp.squeeze(value, axis=-1)

        return pi, value_squeezed
    


def make_ppo_agent(init_rng, config, env_spec: EnvSpec, team_spec: TeamSpec, env, checkpoint_dir, checkpoint_prefix):
    network = SimpleNetwork(env.action_space().n)
    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()
    init_rng, _rng = jax.random.split(init_rng)
    network_params = network.init(_rng, init_x)
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

    init_agent_state = {}
    init_agent_state["eval"] = False
    init_agent_state["train_state"] = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    def get_action(rng, obsv, flattened_obsv, agent_state):
        pi, value = network.apply(agent_state["train_state"].params, flattened_obsv)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        return agent_state, action, (pi, value, log_prob)
    

    # This actually seems to be IPPO loss
    # TODO
    def _loss_fn(net_train_state_params, traj_batch, gae, targets):
        # net_train_state_params = agent_state["train_state"].params
        # gae = Advantages
        # RERUN NETWORK
        flattened_obs = traj_batch.obs.reshape((traj_batch.obs.shape[0], traj_batch.obs.shape[1], -1))
        pi, value = network.apply(net_train_state_params, flattened_obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        # traj_batch.processed_observation[1] == traj_batch.value
        value_pred_clipped = traj_batch.processed_observation[1] + (
            value - traj_batch.processed_observation[1]
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        # CALCULATE ACTOR LOSS
        # traj_batch.processed_observation[2] == traj_batch.log_prob
        ratio = jnp.exp(log_prob - traj_batch.processed_observation[2])
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


    def update(rng, trajectories, agent_state):
        done, actions, aux_data, reward, observations, info = trajectories

        # CALCULATE ADVANTAGE
        # This seems to be the same for PPO and IPPO
        flattened_last_obsv = observations[-1].reshape((observations[-1].shape[0], -1))
        last_val = network.apply(agent_state["train_state"].params, flattened_last_obsv)[1]

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.processed_observation[1],
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.processed_observation[1] # advantages + values

        advantages, targets = _calculate_gae(trajectories, last_val)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, gradients = grad_fn(agent_state["train_state"].params, trajectories, advantages, targets)
        agent_state["train_state"] = agent_state["train_state"].apply_gradients(grads=gradients)

        return agent_state, total_loss
    
    def save(agent_state, step):
        # I guess it's best to pickle and unpickle since the other things don't really seem to work
        # Only save network parameters to save disk space
        target = agent_state["train_state"].params
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, f"{checkpoint_prefix}{step}.param.ckpt"), 'wb') as f:
            pickle.dump(target, f)
        return agent_state

    def load(agent_state, step):
        restored_params = None
        with open(os.path.join(checkpoint_dir, f"{checkpoint_prefix}{step}.param.ckpt"), 'rb') as f:
            restored_params = pickle.load(f)

        new_agent_state = {}
        new_agent_state["eval"] = True
        new_agent_state["train_state"] = TrainState.create(
            apply_fn=network.apply,
            params=restored_params,
            tx=tx
        )
        return new_agent_state
    
    return SelfPlayAgent(
        get_action,
        update,
        save,
        load,
    ), init_agent_state
    






def main():
    job_id = "0"
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    os.makedirs(os.path.join(".", "out", job_id), exist_ok=True)



    config = {
        "CHECKPOINT_DIR": os.path.join(".", "out", job_id, "checkpoints"),
        "ENV_STEPS": 1e2,
        "NUM_UPDATES": 10,
        "NUM_EPISODES": 2,
        # "ANNEAL_LR": True,
        "MAX_GRAD_NORM": 0.5,
        "LR": 2.5e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        # "NUM_ENVS": 3, # 200
        # "NUM_AGENTS": 3, # 32
        # "NUM_STEPS": 25
    }



    rng = jax.random.PRNGKey(0)
    numpy_seed = 0

    # This is part of the config
    # All environments specified here must have the same action space and observation space dimensions
    env_spec = EnvSpec("overcooked", 50, {"layout" : overcooked_layouts["cramped_room"]})
    teams = [ TeamSpec(make_ppo_agent, 4, ['agent_0', 'agent_1']), ]

    jit_device = jax.devices('cpu')[0]
    rng, _rng = jax.random.split(rng)
    stage_1_jit = FCP.make_stage_1(jit_device, config, env_spec, teams, numpy_seed)
    # stage_1_jit = jax.jit(FCP.make_stage_1(config, env_mapping, numpy_seed))
    start_time = datetime.now()
    episode_metrics, last_episode_runner_state = stage_1_jit(_rng)
    stop_time = datetime.now()
    print(f"Elapsed {stop_time-start_time}")


    total_update_steps = config["NUM_UPDATES"] * config["NUM_EPISODES"]
    for team_ix, team_metrics in episode_metrics.items():
        for p_ix, partner_metrics in team_metrics.items():
            plt.plot(range(total_update_steps), partner_metrics[0], label=f"{team_ix}-{p_ix}")
    plt.legend()
    plt.savefig(f"./out/{job_id}/loss_per_partner.png")



    env_spec = EnvSpec("overcooked", 1, {"layout" : overcooked_layouts["cramped_room"]})
    teams = [ TeamSpec(make_ppo_agent, 1, ['agent_0', 'agent_1']), ]
    checkpoint_load_steps = [(config["NUM_EPISODES"]*config["NUM_UPDATES"])-1, ]
    state_seq = get_rollout(config, checkpoint_load_steps, env_spec, teams, max_steps=300)
    env = jaxmarl.make("overcooked", **{"layout" : overcooked_layouts["cramped_room"]})
    viz =  OvercookedVisualizer()
    viz.animate(state_seq, env.agent_view_size, filename=f'./out/{job_id}/animation.gif')

    return episode_metrics, last_episode_runner_state

    team_fcp_agents = [make_ppo_agent, ]
    rng, _rng = jax.random.split(rng)
    stage_2_jit = _make_stage_2(config, env_spec, teams, partners, team_fcp_agents, numpy_seed)
    stage_2_jit(_rng)


if __name__ == "__main__":
    main()
