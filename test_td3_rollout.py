import sys
import os

if __name__ == "__main__":
    if sys.argv[-1] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jaxmarl
import optax

from td3.rollout import make_rollout, visualise
from td3.td3 import DefaultActor
from util.util import pickle_load
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper


def main():
    num_steps = int(sys.argv[1])
    actor_params_path = sys.argv[2]
    out_path = sys.argv[3]

    env = jaxmarl.make("ant_4x2")
    env = LogWrapper(env, replace_info=True)

    config = {
        "NUM_STEPS": num_steps,
        "NUM_ENVS": 25,
        "MAX_ACTION": env.action_spaces[env.agents[0]].high
    }
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    net_actor = DefaultActor(env.action_space(env.agents[0]).shape[0], config["MAX_ACTION"])
    actor_params = pickle_load(actor_params_path)
    state_actor = TrainState.create(
        apply_fn=net_actor.apply,
        params=actor_params,
        tx=optax.adam(3e-4)
    )

    

    rng = jax.random.PRNGKey(0)
    states = make_rollout(
        config, env
    )(rng, state_actor)

    visualise(env, states, config["NUM_STEPS"], out_path)


if __name__ == "__main__":
    main()