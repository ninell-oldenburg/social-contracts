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
    priority: float
    item: Any=field(compare=False)
    

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, env: dm_env.Environment) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """
    self._max_depth = 10
    self.action_spec = env.action_spec()[0]

    # move actions
    self.action_to_pos = [
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
    
    self.rules = Rules()
    
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
    queue.put(PrioritizedItem(0.0, (timestep, plan))) # ordered by reward
    step_count = 0
    prev_reward = timestep.reward

    while not queue.empty():
      priority_item = queue.get()
      cur_timestep, cur_plan = priority_item.item
      if self.is_goal(cur_timestep, prev_reward, step_count):
        return cur_plan[1:] # 'plan' is initialized with a non-empty onset

      # Get the list of actions that are possible and satisfy the rules
      available_actions = self.available_actions(cur_timestep)

      for action in available_actions: # currently exclude all beams
        cur_plan_copy = deepcopy(cur_plan)
        cur_timestep_copy = deepcopy(cur_timestep)
        next_plan =  np.append(cur_plan_copy, action)
        next_timestep = self.env_step(cur_timestep_copy, action)
        queue.put(PrioritizedItem(next_timestep.reward*(-1), # ascending
                                 (next_timestep, next_plan))
                                 )

      step_count += 1

    return False

  def available_actions(self, timestep: dm_env.TimeStep) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    for action in range(7): # self.action_spec.num_values
      if self.is_allowed(timestep, action):
        actions.append(action)

    return actions
  
  def is_allowed(self, timestep, action):
    """Return True if an action is allowed given the current timestep"""
    observation = deepcopy(timestep.observation)
    orientation = observation['ORIENTATION'].item()
    if action <= 4: # if it is a move, alter the position to be observed
      observation['POSITION'] += self.action_to_pos[orientation][action]
    observation = self.update_observation(observation)
    return self.rules.check(observation)
  
  def update_observation(self, observation):
    """Updates the observation with requested information."""
    # lua is 1-indexed
    x, y = observation['POSITION'][0]-1, observation['POSITION'][1]-1
    observation['num_apples_around'] = self.get_apples(observation, x, y)
    observation['has_apple'] = True if observation['SURROUNDINGS'][x][y] == 1 else False
    return observation
  
  def get_apples(self, observation, x, y):
    """Returns the sum of apples around a certain position."""
    sum = 0
    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if observation['SURROUNDINGS'][i][j] == 1:
          sum += 1
    
    return sum

  def is_goal(self, timestep, prev_reward, step_count):
    """Check whether any of the stop criteria are met."""
    if timestep.reward > prev_reward:
      return True
    elif timestep.last():
      return True
    elif step_count == self._max_depth:
      return True
    return False

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()


"""
{
GLOBAL:
'Y', # number of agents that's supposed to be cleaning = 50% of total number of agents
'cleaning_rhythm', # INT
'num_cleaners', # INT
'water_polluted', # BOOL
'cleaner_cleans', # BOOL

AGENT OBSERVATION:
'since_last_cleaned' # INT
'agent_has_stolen', # BOOL
'paid_by_farmer', # BOOL
'cleaner_role', # BOOL
'farmer_role', # BOOL
'apples_paid', # INT
'clean_action', # action

CELL OBSERVATION:
'num_apples_around', # INT
'forgein_property', # BOOL
'has_apples', # BOOL
}
"""