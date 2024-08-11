from typing import Optional

import jax
import jax.experimental

from ficticious_coplay.common import (EnvSpec, SelfPlayAgentFactory, TeamSpec)
from ficticious_coplay.stage_1 import make_stage_1
from ficticious_coplay.stage_2 import make_stage_2


class FCP:
    @staticmethod
    def make_stage_1(config, env_spec: EnvSpec, teams: list[TeamSpec]):
        return jax.jit(make_stage_1(config, env_spec, teams))

    @staticmethod
    def make_stage_2(
        config, env_spec: EnvSpec, teams: list[TeamSpec],
        cls_team_fcp_agents: list[Optional[SelfPlayAgentFactory]],
        checkpoint_load_steps: list[int]
        ):
        return jax.jit(make_stage_2(config, env_spec, teams, cls_team_fcp_agents, checkpoint_load_steps))