from typing import Optional

import jax
import jax.experimental

from ficticious_coplay.common import (EnvSpec, SelfPlayAgentFactory, TeamSpec)
from ficticious_coplay.stage_1 import make_stage_1
from ficticious_coplay.stage_2 import make_stage_2


class FCP:
    @staticmethod
    def make_stage_1(
        config, init_rng: jax.dtypes.prng_key, env_spec: EnvSpec, teams: list[TeamSpec],
        metric_reward_milestone_map: dict[str, int] # Allows us to count the occurence of specific 'milestones' that are attached to a unique reward value
        ):
        return jax.jit(
            make_stage_1(config, init_rng, env_spec, teams, metric_reward_milestone_map),
            device=jax.devices(config["ENV_DEVICE"])[0]
        )

    @staticmethod
    def make_stage_2(
        config, init_rng: jax.dtypes.prng_key, env_spec: EnvSpec, teams: list[TeamSpec],
        cls_team_fcp_agents: list[Optional[SelfPlayAgentFactory]],
        checkpoint_load_steps: list[int],
        metric_reward_milestone_map: dict[str, int] # Allows us to count the occurence of specific 'milestones' that are attached to a unique reward value
        ):
        return jax.jit(
            make_stage_2(config, init_rng, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps, metric_reward_milestone_map),
            device=jax.devices(config["ENV_DEVICE"])[0]
        )