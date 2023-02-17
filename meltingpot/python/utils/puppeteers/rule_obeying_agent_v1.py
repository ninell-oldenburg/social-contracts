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

from collections.abc import Mapping

import dm_env
import numpy as np
import tree

from meltingpot.python.utils.puppeteers import puppeteer

@dataclasses.dataclass(frozen=True)
class ConditionalCleanerState:
  """Current state of the ConditionalCleaner.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    clean_until: step_count after which to stop cleaning.
    recent_cleaning: number of others cleaning on previous timesteps (ordered
      from oldest to most recent).
  """
  step_count: int
  recent_cleaning: tuple[int, ...]
  recently_paid: tuple[bool, ...]

  STATES = {
            'cell.apple': self.state.observation['cell']['apple'], # bool
            'cell.water': self.state.observation['cell']['water'], # 'polluted' / 'clean'
            'global.steps': self.state.observation['global']['steps'], # step count
            'agent.my_property': self.state.observation['agent']['my_property'], # bool
            'agent.enemmies': self.state.observation['agent']['enemies'], # list of enemies
            'agent.role': self.state.observation['agent']['role'], # 'farmer' / 'cleaner' / 'free'
            'agent.paid_by_farmer': self.state.observation['agent']['paid'], # bool
            'agent.clean_actions': self.state.obersvation['agent']['clean_actions'] # list
            }

Observation = Mapping[str, tree.Structure[np.ndarray]]

@dataclasses.dataclass(frozen=True)
class ConditionalFarmerState:
  """Current state of the ConditionalFarmer.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    clean_until: step_count after which to stop cleaning.
    recent_cleaning: number of others cleaning on previous timesteps (ordered
      from oldest to most recent).
  """
  # recent_cleaning: int or bool?
  pays_cleaner: tuple[bool, ...]

@dataclasses.dataclass(frozen=True)
class EgalitarianAgentState:
  """Current state of the EgalitarianAgent.

  Attributes:
    step_count: number of timesteps previously seen in this episode.
    clean_until: step_count after which to stop cleaning.
  """
  step_count: int
  clean_until: int

class RuleObeyingCleaner(puppeteer.Puppeteer[ConditionalCleanerState]):
    def __init__(self,
               *,
               clean_goal: puppeteer.PuppetGoal,
               eat_goal: puppeteer.PuppetGoal,
               coplayer_cleaning_signal: str,
               recency_window: int) -> None:
        """Initializes the puppeteer.

        Args:
        clean_goal: goal to emit to puppet when "cleaning".
        eat_goal: goal to emit to puppet when "eating".
        coplayer_cleaning_signal: key in observations that provides the
            privileged observation of number of others cleaning in the previous
            timestep.
        recency_window: number of steps over which to remember others' behavior.
        """

        self._clean_goal = clean_goal
        self._eat_goal = eat_goal
        self._coplayer_cleaning_signal = coplayer_cleaning_signal

        if recency_window > 0:
            self._recency_window = recency_window
        else:
            raise ValueError('recency_window must be positive')


    def step(self, 
            timestep: dm_env.TimeStep, 
            prev_state: ConditionalCleanerState
            ) -> tuple[dm_env.TimeStep, ConditionalCleanerState]:
        """See base class."""
        if timestep.first():
            prev_state = self.initial_state()
        step_count = prev_state.step_count
        recent_cleaning = prev_state.recent_cleaning
        recently_paid = prev_state.recently_paid

        coplayers_cleaning = int(
            timestep.observation[self._coplayer_cleaning_signal])
        recent_cleaning += (coplayers_cleaning,)
        recent_cleaning = recent_cleaning[-self._recency_window:]
        is_being_paid = bool(
            timestep.observation[self._is_being_paid_signal]
        )
        recently_paid += (is_being_paid,)
        recently_paid = recently_paid[-self._recency_window:]

        """Switch goal to go eat if it is not being paid."""
        if recently_paid:
            goal = self._clean_goal
        else:
            goal = self._eat_goal
        timestep = puppeteer.puppet_timestep(timestep, goal)

        next_state = ConditionalCleanerState(
            step_count=step_count + 1,
            recent_cleaning=recent_cleaning,
            recently_paid=recently_paid)
        return timestep, next_state

    def initial_state(self) -> ConditionalCleanerState:
        return ConditionalCleanerState(
            step_count=0,
            recent_cleaning=(),
            recently_paid=())

class RuleObeyingFarmer(puppeteer.Puppeteer[ConditionalFarmerState]):
    def __init__(self,
               *,
               consume_goal: puppeteer.PuppetGoal,
               pay_goal: puppeteer.PuppetGoal,
               collect_goal: puppeteer.PuppetGoal,
               recency_window: int) -> None:
        """Initializes the puppeteer.

        Args:
        consume_goal: goal to emit to puppet when "consumes".
            Will be emitted when either refined or unrefined tokens
            are in INVENTORY.
        pay_goal: goal to emit to puppet when "pays" others.
            Will be emitted when no refined or double refined tokens
            are in INVENTORY.
        collect_goal: goal to emit to puppet when "collect" apples.
            Will be emitted if INVETORY is empty.
        recency_window: number of steps over which to remember others' behavior.
        """

        self._consume_goal = consume_goal
        self._pay_goal = pay_goal
        self._collect_goal = collect_goal

        if recency_window > 0:
            self._recency_window = recency_window
        else:
            raise ValueError('recency_window must be positive')

    def step(self, 
            timestep: dm_env.TimeStep, 
            prev_state: ConditionalFarmerState
            ) -> tuple[dm_env.TimeStep, ConditionalFarmerState]:
        """See base class."""
        if timestep.first():
            prev_state = self.initial_state()
        pays_cleaner = prev_state.pays_cleaner

        is_paying_cleaner = bool(
            timestep.observation[self._is_paying_signal]
        )
        pays_cleaner += (is_paying_cleaner,)
        pays_cleaner = pays_cleaner[-self._recency_window:]

        if np.sum(timestep.observation['INVENTORY']):
            if self.should_consume(timestep.observation):
                goal = self._consume_goal
            else:
                goal = self._pay_goal
        else:
            goal = self._collect_goal
        timestep = puppeteer.puppet_timestep(timestep, goal)

        next_state = ConditionalFarmerState(
            pays_cleaner=pays_cleaner)
        return timestep, next_state

    def initial_state(self) -> ConditionalFarmerState:
        return ConditionalFarmerState(
            recent_cleaning=(),
            pays_cleaner=())

    def should_consume(self, observation: Observation) -> bool:
        """Decides whether we should consume tokens in our inventory."""
        _, refined, twice_refined = observation['INVENTORY']
        return bool(refined) or bool(twice_refined)

class RuleObeyingEgalitarian(puppeteer.Puppeteer[EgalitarianAgentState]):
    def __init__(self,
               *,
               clean_goal: puppeteer.PuppetGoal,
               eat_goal: puppeteer.PuppetGoal,
               recency_window: int,
               cleaning_period: int) -> None:
        """Initializes the puppeteer.

        Args:
        clean_goal: goal to emit to puppet when "cleaning".
        eat_goal: goal to emit to puppet when "eating".
        recency_window: number of steps over which to remember others' behavior.
        """

        self._clean_goal = clean_goal
        self._eat_goal = eat_goal

        if cleaning_period > 0:
            self._cleaning_period = cleaning_period
        else:
            raise ValueError('cleaning_period must be positive')

        if recency_window > 0:
            self._recency_window = recency_window
        else:
            raise ValueError('recency_window must be positive')

    def step(self, 
             timestep: dm_env.TimeStep, 
             prev_state: EgalitarianAgentState
             ) -> tuple[dm_env.TimeStep, EgalitarianAgentState]:

        if timestep.first():
            prev_state = self.initial_state()
        step_count = prev_state.step_count
        clean_until = prev_state.clean_until

        clean_until = max(clean_until, step_count + self._cleaning_period)

        if step_count < clean_until:
            goal = self._clean_goal
        else:
            goal = self._eat_goal
        timestep = puppeteer.puppet_timestep(timestep, goal)

        next_state = EgalitarianAgentState(
            step_count=step_count+1,
            clean_until=clean_until
        )

        return timestep, next_state

    def initial_state(self) -> EgalitarianAgentState:
        return EgalitarianAgentState(
            step_count=0,
            clean_until=20
        )