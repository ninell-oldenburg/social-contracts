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

from typing import Tuple

import dm_env
import dmlab2d

from dataclasses import dataclass, field
from typing import Any

from queue import PriorityQueue

import numpy as np
from copy import deepcopy

from meltingpot.python.utils.policies import policy

# https://github.com/deepmind/meltingpot/blob/main/examples/rllib/utils.py

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, env: dm_env.Environment) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """
    self._max_depth = 20
    self._env = env
    self.action_spec = env.action_spec()[0]
    self.observation_spec = env.observation_spec()

    # move actions
    self.action_to_position = [
            [[0,0],[0,-1],[0,1],[-1,0],[1,0]], # N
            [[0,0],[1,0],[-1,0],[0,1],[0,-1]], # E
            [[0,0],[0,1],[0,-1],[1,0],[-1,0]], # S
            [[0,0],[-1,0],[1,0],[0,-1],[0,1]], # W
          ]
    
    # turn actions
    self.action_to_orientation = [
            [3, 1], # N
            [0, 2], # E
            [1, 3], # S
            [2, 0], # W
          ]
    
  def step(self, 
           timestep: dm_env.TimeStep,
           ) -> Tuple[int, policy.State]:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """

      # Select an action based on the first satisfying rule
      action_plan = self.a_star(timestep)

      return action_plan
  
  def get_reward(self, observation) -> float:
    # lua is one indexed
    x, y = observation['POSITION'][0]-1, observation['POSITION'][1]-1
    reward_map = observation['SURROUNDINGS']
    return reward_map[x][y]

  def env_step(self, timestep: dm_env.TimeStep, action) -> dm_env.TimeStep:
      # Unpack observations from timestep
      observation = timestep.observation
      orientation = observation['ORIENTATION']
      reward = timestep.reward
      # Simulate changes to observation based on action
      # move actions
      if action <= 4:
        observation['POSITION'] += self.action_to_position[orientation.item()][action]
        reward += self.get_reward(observation)
      # turn actions
      elif action <= 6:
        action = action - 5 # indexing starts at 0
        observation['ORIENTATION'] = np.array(self.action_to_orientation[orientation.item()][action])
      # TODO implement FIRE_ZAP, FIRE_CLEAN, FIRE_CLAIM
      else:
        pass

      return dm_env.TimeStep(step_type=dm_env.StepType.MID,
                                     reward=reward,
                                     discount=1.0,
                                     observation=observation,
                                     )

      
  def a_star(self, timestep: dm_env.TimeStep) -> list[int]:
    """Perform a a_star search to generate plan."""
    plan = np.zeros(shape=1, dtype=int)
    queue = PriorityQueue()
    queue.put(PrioritizedItem(0.0, (timestep, plan))) # ordered by reward
    step_count = 0
    prev_reward = timestep.reward

    while not queue.empty():
      priority_item = queue.get()
      cur_timestep, cur_plan = priority_item.item
      # goal: increase reward in comparison to previous timestep
      if self.is_goal(cur_timestep, prev_reward):
        # np array can't be empty so we're initializing it with 1 digit
        # and don't return that first digit
        return cur_plan[1:]
      elif cur_timestep.last() or step_count == self._max_depth:
        return cur_plan[1:]

      # Get the list of actions that are possible and satisfy the rules
      # available_actions = self.available_actions(cur_timestep)

      # iter over actions 
      for action in range(7):
        # copy plan and timestep
        cur_plan_copy = deepcopy(cur_plan)
        cur_timestep_copy = deepcopy(cur_timestep)

        # create new plan and timestep
        next_plan =  np.append(cur_plan_copy, action)
        next_timestep = self.env_step(cur_timestep_copy, action)
        queue.put(PrioritizedItem(next_timestep.reward*(-1), (next_timestep, next_plan)))

      step_count += 1

    return False

  def available_actions(self, timestep: dm_env.TimeStep) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    for action in range(self.action_spec.num_values):
      # TODO: implement actual logic via pySMT
      is_allowed = self.check_rules(timestep, action)
      if is_allowed:
        actions.append(action)

    return actions
  
  def check_rules(self, timestep, action):
    # Check if action is allowed according to all the rules, given the current timestep
    for rule in self.rules:
      pass
    return True

  def is_goal(self, timestep, prev_reward):
    # Check if agent has reached an apple
    if timestep.reward > prev_reward:
      return True
    return False

  
  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

"""

class CustomPriorityQueue(PriorityQueue):
    def __init__(self,tuple):
      PriorityQueue.__init__(self)
      super()._put((self._get_priority(tuple), tuple))

    def _put(self, tuple):
      return super()._put((self._get_priority(tuple), tuple))
    
    def _get(self):
      return super()._get()[1]

    def _get_priority(self, tuple):
      return tuple[0].reward"""
