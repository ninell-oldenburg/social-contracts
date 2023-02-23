# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Game rules implementation in first-order logic."""

import dataclasses

import dm_env

from meltingpot.python.utils.puppeteers import puppeteer

@dataclasses.dataclass(frozen=True)
class RuleObeyingAgentState:
  """Current state of the RuleObeyingAgent.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    clean_until: step_count after which to stop cleaning.
    recent_cleaning: number of others cleaning on previous timesteps (ordered
      from oldest to most recent).
  """

  # we might not want to have all of those hard coded parameters
  step_count: int
  clean_until: int
  recent_cleaning: tuple[int,...] # others cleaning
  recent_self_cleaning: tuple[bool,...] # has cleaned in previous steps
  recently_paid: tuple[bool,...]

class RuleObeyingAgent():
    def __init__(self) -> None:
        pass
            
    def step(self, 
            action_simluation,
            state,
            observations,
            prev_reward) -> tuple[dm_env.TimeStep, RuleObeyingAgentState]:
        """See base class."""

        step_type = dm_env.StepType.MID
        reward = 0
        discount = 0.1
        observation = observations

        new_timestep = dm_env.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=[observation],
        )

        new_state = RuleObeyingAgentState(
            step_count=state.step_count + 1,
            clean_until=state.clean_until,
            recent_cleaning=state.recent_cleaning,
            recent_self_cleaning=state.recent_self_cleaning,
            recently_paid=state.recently_paid,
            )

        return new_timestep, new_state
    
    def initial_state(self) -> RuleObeyingAgentState:
        return RuleObeyingAgentState(
            step_count=0,
            clean_until=20,
            recent_cleaning=(),
            recent_self_cleaning=(),
            recently_paid=(),
            )
