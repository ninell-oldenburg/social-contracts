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

from dataclasses import dataclass, field
from typing import Any

from queue import PriorityQueue

from meltingpot.python.utils.policies.pysmt_rules import Rules

import numpy as np
from copy import deepcopy

from meltingpot.python.utils.policies import policy

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    tie_break: float
    item: Any=field(compare=False)
    

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, env: dm_env.Environment, components: list) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """
    self.components = components
    self._max_depth = 4
    self.action_spec = env.action_spec()[0]

    # move actions
    self.action_to_pos = [
            [[0,0],[0,-1],[0,1],[-1,0],[1,0]], # N
            [[0,0],[1,0],[-1,0],[0,-1],[0,1]], # E
            [[0,0],[0,1],[0,-1],[1,0],[-1,0]], # S
            [[0,0],[-1,0],[1,0],[0,1],[0,-1]], # W
            # N    # F    # SR  # B   # SL
          ]
    # turn actions
    self.action_to_orientation = [
            [3, 1], # N
            [0, 2], # E
            [1, 3], # S
            [2, 0], # W
          ]
    # beams
    self.action_to_beam = [
            "CLEAN_ACTION",
            "ZAP_ACTION",
            "CLAIM_ACTION",
          ]
    
    self.rules = Rules(components)
    
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
    if self.exceeds_map(observation['WORLD.RGB'], x, y):
      return (-1)
    return reward_map[x][y]

  def env_step(self, timestep: dm_env.TimeStep, action) -> dm_env.TimeStep:
      # Unpack observations from timestep
      observation = timestep.observation
      orientation = observation['ORIENTATION'].item()
      reward = timestep.reward

      # Simulate changes to observation based on action
      if action <= 4: # move actions
        observation['POSITION'] += self.action_to_pos[orientation][action]
        reward += self.get_reward(observation)
      
      elif action <= 6: # turn actions
        action = action - 5 # indexing starts at 0
        observation['ORIENTATION'] = np.array(self.action_to_orientation
                                             [orientation][action]
                                             )
      # TODO implement FIRE_ZAP, FIRE_CLEAN, FIRE_CLAIM
      else:
        pass

      return dm_env.TimeStep(step_type=dm_env.StepType.MID,
                                     reward=reward,
                                     discount=1.0,
                                     observation=observation,
                                     )

      
  def a_star(self, timestep: dm_env.TimeStep) -> list[int]:
    """Perform a A* search to generate plan."""
    plan = np.zeros(shape=1, dtype=int)
    queue = PriorityQueue()
    timestep = timestep._replace(reward=0.0) # inherits from calling call
    queue.put(PrioritizedItem(0.0, 0, (timestep, plan))) # ordered by reward

    while not queue.empty():
      priority_item = queue.get()
      cur_timestep, cur_plan = priority_item.item
      if self.meets_break_criterium(cur_timestep, len(cur_plan)):
        return cur_plan[1:] # 'plan' is initialized with a non-empty onset

      # Get the list of actions that are possible and satisfy the rules
      available_actions = self.available_actions(cur_timestep)

      for action in available_actions: # currently exclude all beams
        cur_plan_copy = deepcopy(cur_plan)
        cur_timestep_copy = deepcopy(cur_timestep)
        next_plan =  np.append(cur_plan_copy, action)
        next_timestep = self.env_step(cur_timestep_copy, action)
        queue.put(PrioritizedItem(priority=len(next_plan),
                                  tie_break=next_timestep.reward*(-1), # ascending
                                  item=(next_timestep, next_plan))
                                 )

    return False

  def available_actions(self, timestep: dm_env.TimeStep) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    for action in range(self.action_spec.num_values):
      if self.is_allowed(timestep, action):
        actions.append(action)

    return actions
  
  def is_allowed(self, timestep, action):
    """Returns True if an action is allowed given the current timestep"""
    observation = deepcopy(timestep.observation)
    orientation = observation['ORIENTATION'].item()
    if action <= 4: # record and alter move
      observation['POSITION'] += self.action_to_pos[orientation][action]
    elif action >= 7: # record and alter beam
      action = action - 7 # zero-indexed
      action_name = self.action_to_beam[action]
      observation[action_name] = True # to check for pySMT rules
    
    observation = self.update_observation(observation)
    return self.rules.check_all(observation)
  
  def update_observation(self, observation):
    # TODO: make this readable
    """Updates the observation with requested information."""
    # lua is 1-indexed
    x, y = observation['POSITION'][0]-1, observation['POSITION'][1]-1
    observation['NUM_APPLES_AROUND'] = self.get_apples(observation, x, y)
    observation['CUR_CELL_HAS_APPLE'] = True if not self.exceeds_map(observation['WORLD.RGB'], x, y) \
      and observation['SURROUNDINGS'][x][y] == 1 else False
    if 'pollution' in self.components:
      observation['IS_AT_WATER'] = True if observation['IS_AT_WATER'] == 1 else False
    for action in self.action_to_beam:
        if action not in observation.keys(): observation[action] = False 
        
    return observation
  
  def get_apples(self, observation, x, y):
    """Returns the sum of apples around a certain position."""
    sum = 0
    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if not self.exceeds_map(observation['WORLD.RGB'], i, j):
          if observation['SURROUNDINGS'][i][j] == 1:
            sum += 1
    
    return sum
  
  def exceeds_map(self, world_rgb, x, y):
    x_max = world_rgb.shape[1] / 8
    y_max = world_rgb.shape[0] / 8
    if x <= 0 or x >= x_max:
      return True
    if y <= 0 or y >= y_max:
      return True
    return False

  def meets_break_criterium(self, timestep, plan_length):
    """Check whether any of the stop criteria are met."""
    if timestep.reward >= 1.0:
      return True
    elif timestep.last():
      return True
    elif plan_length > self._max_depth:
      return True
    return False

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
