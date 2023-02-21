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
  # TODO: implement matrix of who's eaten what
  #enemies: list[str] # list of people who've eaten from your property
  #role: str


class RuleObeyingAgent(puppeteer.Puppeteer[RuleObeyingAgentState]):
    def __init__(self) -> None:
        """Initializes the puppeteer.

        Args:
        coplayer_cleaning_signal: key in observations that provides the
            privileged observation of number of others cleaning in the previous
            timestep.
        recency_window: number of steps over which to remember others' behavior.
        threshold: if the total number of (nonunique) cleaners over the
            remembered period reaches this threshold, the puppeteer will direct the
            puppet to clean.
        reciprocation_period: the number of steps to clean for once others'
            cleaning has been forgotten and fallen back below threshold.
        

        self._coplayer_cleaning_signal = coplayer_cleaning_signal

        if threshold > 0:
            self._threshold = threshold
        else:
            raise ValueError('threshold must be positive')

        if recency_window > 0:
            self._recency_window = recency_window
        else:
            raise ValueError('recency_window must be positive')

        if reciprocation_period > 0:
            self._reciprocation_period = reciprocation_period
        else:
            raise ValueError('reciprocation_period must be positive')
        """
            
    def step(self, 
            timestep: dm_env.TimeStep, 
            prev_state: RuleObeyingAgentState 
            ) -> tuple[dm_env.TimeStep, RuleObeyingAgentState]:
        """See base class."""
        if timestep.first():
            prev_state = self.initial_state()
        step_count = prev_state.step_count
        clean_until = prev_state.clean_until
        recent_cleaning = prev_state.recent_cleaning
        recent_self_cleaning= prev_state.recent_self_cleaning
        recently_paid = prev_state.recently_paid
        #enemies=prev_state.enemies
        #role=prev_state.role

        """
        # Did coplayers clean recently
        coplayers_cleaning = int(
            timestep.observation[self._coplayer_cleaning_signal])
        recent_cleaning += (coplayers_cleaning,)
        recent_cleaning = recent_cleaning[-self._recency_window:]

        # Did I clean recently
        myself_cleaning = bool(
            timestep.observation[self._myself_cleaning_signal]
        )
        recent_self_cleaning += (myself_cleaning,)
        recent_self_cleaning = recent_self_cleaning[-self._recency_window:]

        # Was I paid recently?
        is_being_paid = bool(
            timestep.observation[self._is_being_paid_signal]
        )
        recently_paid += (is_being_paid,)
        recently_paid = recently_paid[-self._recency_window:]

        # Calculate step count until I clean
        smooth_cleaning = sum(recent_cleaning)
        if smooth_cleaning >= self._threshold:
            clean_until = max(clean_until, step_count + self._reciprocation_period)
        # Do not clear the recent_cleaning history after triggering.
        """

        next_state = RuleObeyingAgentState(
            step_count=step_count + 1,
            clean_until=clean_until,
            recent_cleaning=recent_cleaning,
            recent_self_cleaning=recent_self_cleaning,
            recently_paid=recently_paid,
            #enemies=enemies,
            #role=role
            )

        return next_state
    
    def initial_state(self) -> RuleObeyingAgentState:
        return RuleObeyingAgentState(
            step_count=0,
            clean_until=20,
            recent_cleaning=(),
            recent_self_cleaning=(),
            recently_paid=(),
            #enemies=[],
            #role=''
            )
