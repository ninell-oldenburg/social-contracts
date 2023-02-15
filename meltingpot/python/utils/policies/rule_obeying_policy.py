# Copyright 2020 DeepMind Technologies Limited.
#
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
"""Bot policy implementations."""

from typing import Generic, Tuple, TypeVar

import dm_env
import dmlab2d

from meltingpot.python.utils.policies import policy
# what's puppet and puppeteer?
from meltingpot.python.utils.substrates import commons_harvest_rules as rules

PuppetState = TypeVar('PuppetState')
# https://github.com/deepmind/meltingpot/blob/main/examples/rllib/utils.py

class RuleObeyingPolicy(policy.State):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, 
               puppet: policy.State,
               env: dmlab2d.Environment) -> None:
    """Initializes the policy.

    Args:
      puppet: The puppet policy. Will be closed with this wrapper.
    """
    self._agent_state = puppet
    self._rules = rules
    self._env = env

  def available_actions(self, 
                        timestep: dm_env.TimeStep, 
                        state: policy.State
                        ) -> list:
    actions = []
    for _ in len(rules):
      if timestep.last():
        return []
      if self._rules[state]:
        actions.append((timestep, self._rules[state]))

    """Sort actions by reward."""
    actions = sorted(actions, key=lambda action : action[1].reward) 
    return actions

  """
  observation_spec: Dict(
  NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP:Box(-inf, inf, (), float64), 
  NUM_OTHERS_WHO_ATE_THIS_STEP:Box(-inf, inf, (), float64), 
  NUM_OTHERS_WHO_CLEANED_THIS_STEP:Box(-inf, inf, (), float64), 
  ORIENTATION:Box(-2147483648, 2147483647, (), int32), 
  PLAYER_ATE_APPLE:Box(-inf, inf, (), float64), 
  PLAYER_CLEANED:Box(-inf, inf, (), float64), 
  POSITION:Box(-2147483648, 2147483647, (2,), int32), 
  READY_TO_SHOOT:Box(-inf, inf, (), float64), 
  RGB:Box(0, 255, (88, 88, 3), uint8))
  

  action_spec: Discrete(9)
  """

  def forward_bfs(self) -> Tuple[int, policy.State]:
      """
      Forward breadth first search.
      """
      plan = []
      state = self.initial_state()
      timestep = self._env.reset()

      # is the timestep calculated internally
      # i.e., do we not need to call it initially?
      queue = [(timestep, state, plan)]
      while queue:
        this_timestep, state, plan = queue.pop(0)
        
        if this_timestep.last():
          # we don't actually need to return a plan here do we?
          return plan

        # assumption: available actions only depend on the state
        avaiable_actions = self.available_actions(state)
        for action in avaiable_actions:
          # again, do we need to explicitly calculate the timestep here?
          next_timestep, next_state = self.step(this_timestep, state, action)
          next_plan =  [plan, action]
          queue.append((next_timestep, next_state, next_plan))

      return False

  def step(self, 
           timestep: dm_env.TimeStep, 
           state: policy.State,
           ) -> Tuple[int, policy.State]:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """
      observations = {
        key: value
        for key, value in timestep.observation.items()
        if 'WORLD' not in key
    }

      next_action, next_state = self._env.step(
        observations,
        state,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)
        
      return next_action, next_state
  
  def initial_state(self) -> Tuple[()]:
    """See base class."""
    return (self._agent.initial_state())

  def close(self) -> None:
    """See base class."""
    self._agent.close()
