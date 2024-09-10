import sys
import os

if __name__ == "__main__":
    if sys.argv[-1] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jaxmarl
import optax

from td3.rollout import _make_rollout, visualise
from td3.td3 import DefaultActor
from util.util import pickle_load
from flax.training.train_state import TrainState


def main():
    num_steps = int(sys.argv[1])
    actor_params_path = sys.argv[2]
    out_path = sys.argv[3]

    env = jaxmarl.make("ant_4x2")
    max_action = env.action_spaces[env.agents[0]].high
    net_actor = DefaultActor(env.action_space(env.agents[0]).shape[0], max_action)
    actor_params = pickle_load(actor_params_path)
    state_actor = TrainState.create(
        apply_fn=net_actor.apply,
        params=actor_params,
        tx=optax.adam(3e-4)
    )

    rng = jax.random.PRNGKey(0)
    states = jax.jit(_make_rollout(env, num_steps))(rng, state_actor)

    visualise(env, states, num_steps, out_path)


if __name__ == "__main__":
    main()