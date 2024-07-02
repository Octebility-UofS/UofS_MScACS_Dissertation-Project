from typing import NamedTuple, Sequence, Type
import flax.linen as nn
from flax import serialization
import jax
import jaxmarl
from jaxmarl.environments import SimpleReferenceMPE
import numpy as np
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax

## ===
# Checkpointing https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
## ===

# Stage 1 - Training Diverse Pool of Parnters
## Train N Partner agents in self-play (independently)
## To represent a range of skill levels, use multiple checkpoints for each partner agent throughout training
## Architecture of partners is the same for each since paper found no real gain
## Therefore, training partners vary only in random init seed

# Stage 2 - Train FCP agent as best response to diverse partner agents
## Partner weights are frozen which forces FCP agent to adapt to them



# Algorithm 1: Fictitious Co-Play (FCP)
# Input: Number of partners N , checkpoint frequency nc, checkpoint filter F
# // Stage 1: train diverse partner population
# partners = []
# for i = 1 to N do
    # Initialize agent i.
    # n = 0 // step count
    # while not converged do
        # Update agent i in self-play.
        # n += 1
        # if n mod nc = 0 then
            # Add frozen agent i checkpoint to partners.
# // Stage 2: train FCP agent
# Filter partners with F .
# Initialize FCP agent.
# while not converged do
    # Sample partner from partners.
    # Update FCP in co-play with partner.


class SelfPlayAgent:

    def train(self):
        """
        Sets the agent to training mode. Gradients are recorded
        """
        raise NotImplementedError()

    def eval(self):
        """
        Sets the agent to evaluation mode. No gradients are recorded
        """
        raise NotImplementedError()

    def get_action(state, local_observation, global_observations):
        raise NotImplementedError()

    def update_epoch(self):
        raise NotImplementedError()

    def _update_minibatch(self):
        raise NotImplementedError()

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

config = {
    "NUM_ENVS": 50, # 200
    "NUM_AGENTS": 8, # 32
    "NUM_STEPS": 25
}

def make_train(config):
    env: SimpleReferenceMPE = jaxmarl.make("MPE_simple_reference_v3")

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = 10
    
    def train(rng):
        # INIT NETWORK
        # At the moment, no network


        # INIT ENV
        rand_key_env_reset_array = jax.random.split(rng, config['NUM_ENVS'])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(rand_key_env_reset_array)

        # TRAIN LOOP
        def  _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                value = jnp.array([1.0, ])
                log_prob = jnp.array([0.0, ])

                # for now, just select random actions for each
                action = jnp.zeros(config["NUM_ENVS"] * env.num_agents, dtype=jnp.int32)
                for i in range(config["NUM_ENVS"]):
                    for ix, agent_id in enumerate(env.agents):
                        action = action.at[i+ix].set(env.action_space(agent_id).sample(_rng))

                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                print(env_act)
                env_act = {k: v.squeeze() for k, v in env_act.items()}
                print(env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, rng)
                return runner_state, transition
            
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            # ac_in = (
            #     last_obs_batch[np.newaxis, :],
            #     last_done[np.newaxis, :],
            # )
            # _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            # last_val = last_val.squeeze()

            # advantages, targets = _calculate_gae(traj_batch, last_val)

            update_state = (
                train_state,
                traj_batch,
                None, #advantages
                None, #targets
                rng,
            )
            # update_state, loss_info = jax.lax.scan(
            #     _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            # )
            train_state = update_state[0]
            rng = update_state[-1]
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, rng)
            metric = None
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            None, #train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train






def main():
    rng = jax.random.PRNGKey(0)

    train_jit = jax.jit(make_train(config), device=jax.devices()[0])
    out = train_jit(rng)
    print(out)

    



if __name__ == "__main__":
    main()

    # agents = [ SelfPlayAgent() for _ in range(NUM_AGENTS) ]


    # env: SimpleReferenceMPE = jaxmarl.make("MPE_simple_reference_v3")
    # rand_key_env_reset_array = jax.random.split(rand_key, NUM_ENVS)
    # env_obsv_array, env_state_array = jax.vmap(env.reset, in_axes=(0,))(rand_key_env_reset_array)

    # for ix_game in range(NUM_GAMES):
    #     # Randomly choose agents for each environment (with replacement)
    #     total_req_agents = env.num_agents * NUM_ENVS
    #     rand_key_agent_sample = jax.random.split(rand_key, 1)
    #     sampled_agents = jax.random.choice(rand_key_agent_sample, agents, shape=(total_req_agents, ), replace=True)

    # print(env.__dict__.keys())
    # observations, state = env.reset(rand_key)
    # print(observations)
    # print(state)
    # actions = {agent_id: env.action_space(agent_id).sample(rand_key) for agent_id in env.agents}
    # obsv, env_state, reward, done, info = env.step(rand_key, state, actions)
    # print(obsv)
    # print(env_state)
    # print(reward)
    # print(done)
    # print(info)
    # 
    # env: SimpleReferenceMPE = get_environment()
    
    # agents = [] # The agent models (Trainstate? because that includes the optimizer)
    # envs = [] # The available environments
    # map_agents_to_envs() # Function that assigns agents to environments at each iteration
    # algorithm # The algorithm that updates the agent networks


    




# def pseudo():
#     rand_key = jax.random.PRNGKey(0)
#     env = jaxmarl.make("env_id")
#     for i in num_envs:
#         rand_key_reset, rand_key_agents = jax.random.split(rand_key, 1)
#         obsv, state = env.reset(rand_key_reset)
#         rand_key_agents_array = jax.random.split(rand_key, env.num_agents)
#         actions = {}
#         for ix, agent_id in enumerate(env.agents):
#             actions[agent_id] = env.action_space(agent_id).sample(rand_key_agents_array[ix])

        



# def train():
#     env = jaxmarl.make("env_id")
#     train_state: TrainState = get_actor_train_state()
    
#     # Initiate environments
#     rng, _rng = jax.random.split(rng)
#     reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
#     obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

    





# # Currently being implemented for fully cooperative games
# # At the moment, all actors must have the same role
# class FCP:
#     def __init__(
#             self, partner_model: Type[nn.Module], agent_model: Type[nn.Module],
#             num_partners, partner_steps, agent_steps,
#             checkpoint_frequency, checkpoint_filter
#         ):
#         self.partner_model = partner_model
#         self.partners = []

#     def train_partners(self):
#         pass

#     def _selfplay(self, partner_model_instance):
#         pass

#     def _coplay(self):
#         pass

#     def _save_model(self, model: nn.Module, file_name=None):
#         raise NotImplementedError("Still need to figure this out")

#     def _load_model(self, model: Type[nn.Module], state_dict=None, file_name=None):
#         raise NotImplementedError("Still need to figure this out")