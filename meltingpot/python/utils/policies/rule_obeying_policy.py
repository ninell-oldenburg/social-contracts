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

from pysmt.shortcuts import *

from meltingpot.python.utils.policies.pysmt_rules import ProhibitionRule, ObligationRule

import numpy as np
from copy import deepcopy

from meltingpot.python.utils.policies import policy

foreign_property = Symbol('CUR_CELL_IS_FOREIGN_PROPERTY', BOOL)
cur_cell_has_apple = Symbol('CUR_CELL_HAS_APPLE', BOOL)
agent_has_stolen = Symbol('AGENT_HAS_STOLEN', BOOL)
clean_action = Symbol('CLEAN_ACTION', BOOL)
pay_action = Symbol('PAY_ACTION', BOOL)
dirt_fraction = Symbol('DIRT_FRACTION', REAL)
cleaner_role = Symbol('CLEANER_ROLE', BOOL)
farmer_role = Symbol('FARMER_ROLE', BOOL)

"""
            OBLIGATION:
            # every X turns, go clean the water
            Implies(Equals(Symbol('since_last_cleaned', INT), Symbol('cleaning_rhythm', INT)),
                    clean_action),
            # if I'm in the cleaner role, go clean the water
            Implies(cleaner_role, clean_action),

            PERMISSION:
            # Stop cleaning if I'm not paid by farmer
            Implies(And(cleaner_role, Not(Symbol('paid_by_farmer', BOOL))), 
                    Not(clean_action)),
            # stop paying cleaner if they don't clean
            Implies(Not(Symbol('cleaner_cleans', BOOL)), Not(pay_action)))
            ]
            """

DEFAULT_OBLIGATIONS = [
    # every time the water gets too polluted, go clean the water
    ObligationRule(GT(dirt_fraction, Real(0.6)), 'CLEAN_ACTION'),
    # clean the water if less than Y agents are cleaning
    ObligationRule(LT(Symbol('NUM_CLEANERS', REAL), Real(1)), 
                    'CLEAN_ACTION'),
    # Pay cleaner with apples
    # ObligationRule(And(farmer_role, Equals(Symbol('since_last_payed', INT),\
                      # Symbol('pay_rhythm', INT))), "PAY_ACTION"),
]

DEFAULT_PROHIBITIONS = [
    # don't go if <2 apples around
    ProhibitionRule(Not(And(cur_cell_has_apple, LT(Symbol('NUM_APPLES_AROUND', INT), Int(3))))),
    # don't fire the cleaning beam if you're not close to the water
    ProhibitionRule(Not(And(clean_action, Not(Symbol('IS_AT_WATER', BOOL))))),
    # don't go if it is foreign property and cell has apples 
    ProhibitionRule(Not(And(Not(agent_has_stolen), And(foreign_property, cur_cell_has_apple)))),
]

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    tie_break: float
    item: Any=field(compare=False)
    

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, 
               env: dm_env.Environment, 
               player_idx: int, 
               prohibitions: list = DEFAULT_PROHIBITIONS, 
               obligations: list = DEFAULT_OBLIGATIONS) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """
    self._index = player_idx
    self._max_depth = 25
    self.action_spec = env.action_spec()[0]
    self.prohibitions = prohibitions
    self.obligations = obligations
    self.current_obligation = None

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
            "ZAP_ACTION",
            "CLEAN_ACTION",
            "CLAIM_ACTION",
            "EAT_ACTION",
            "PAY_ACTION"
          ]
        
  def step(self, 
           timestep: dm_env.TimeStep
           ) -> Tuple[int, policy.State]:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """

      # Check if any of obligations are active
      self.current_obligation = None
      for rule in self.obligations:
         if rule.holds(timestep.observation):
           self.current_obligation = rule
           break
         
      print(f"self.current_obligation == None: {self.current_obligation == None}")

      # Select an action based on the first satisfying rule
      action_plan = self.a_star(timestep)

      return action_plan
  
  def maybe_collect_apple(self, observation) -> float:
    # lua is one indexed
    x, y = observation['POSITION'][0]-1, observation['POSITION'][1]-1
    reward_map = observation['SURROUNDINGS']
    if self.exceeds_map(observation['WORLD.RGB'], x, y):
      return 0
    return reward_map[x][y]

  def env_step(self, timestep: dm_env.TimeStep, action) -> dm_env.TimeStep:
      # Unpack observations from timestep
      observation = timestep.observation
      orientation = observation['ORIENTATION'].item()
      reward = timestep.reward
      cur_inventory = int(observation['INVENTORY'])

      # Simulate changes to observation based on action
      if action <= 4: # move actions
        observation['POSITION'] += self.action_to_pos[orientation][action]
        cur_inventory += self.maybe_collect_apple(observation)
      
      elif action <= 6: # turn actions
        action = action - 5 # indexing starts at 0
        observation['ORIENTATION'] = np.array(self.action_to_orientation
                                             [orientation][action])
        
      else: # beams, pay, and eat actions
        if cur_inventory > 0:
          if action >= 10:
            if action == 10: # eat
              reward += 1 # TODO: change from hard-coded to variable
          cur_inventory -= 1 # pay

      observation['INVENTORY'] = cur_inventory

      return dm_env.TimeStep(step_type=dm_env.StepType.MID,
                                     reward=reward,
                                     discount=1.0,
                                     observation=observation,
                                     )

      
  def reconstruct_path(self, came_from: dict, timestep_action: tuple) -> list:
    path = [timestep_action[1]]
    while timestep_action in came_from.keys():
      if not timestep_action == came_from[timestep_action]:
        timestep_action = came_from[timestep_action]
        path.append(timestep_action[1])
      else:
        break
    path.reverse()
    return path
  
  def a_star(self, timestep: dm_env.TimeStep) -> list[int]:
    """Perform a A* search to generate plan."""
    queue = PriorityQueue()
    action = 0
    came_from = {}
    timestep = timestep._replace(reward=0.0) # inherits from calling timestep
    queue.put(PrioritizedItem(0, 0, (timestep, action))) # ordered by reward

    while not queue.empty():
      priority_item = queue.get()
      cur_timestep, cur_action = priority_item.item
      cur_position = tuple(cur_timestep.observation['POSITION'])
      cur_depth = priority_item.priority
      if self.is_done(cur_timestep, cur_depth, cur_action):
        return self.reconstruct_path(came_from, (cur_position, cur_action))

      # Get the list of actions that are possible and satisfy the rules
      available_actions = self.available_actions(cur_timestep)

      for action in available_actions:
        cur_timestep_copy = deepcopy(cur_timestep)
        next_timestep = self.env_step(cur_timestep_copy, action)
        next_position = tuple(next_timestep.observation['POSITION'])
        if not (next_position, action) in came_from.keys() \
          or next_timestep.reward > cur_timestep.reward:
          came_from[(next_position, action)] = (cur_position, cur_action)
          queue.put(PrioritizedItem(priority=cur_depth+1,
                                    tie_break=next_timestep.reward*(-1), # ascending
                                    item=(next_timestep, action))
                                    )
    return False

  def available_actions(self, timestep: dm_env.TimeStep) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    observation = deepcopy(timestep.observation)
    orientation = observation['ORIENTATION'].item()

    for action in range(self.action_spec.num_values):
      if action <= 4: # move actions
        observation['POSITION'] += self.action_to_pos[orientation][action]

      x, y = observation['POSITION'][0]-1, observation['POSITION'][1]-1
      if self.exceeds_map(observation['WORLD.RGB'], x, y):
        continue
      
      if self.current_obligation != None:
        if not action in self.current_obligation.get_valid_actions():
          continue

      observation = self.update_observation(observation, action, x, y)
      if not self.check_all(observation):
        continue
    
      actions.append(action)
    return actions
  
  def check_all(self, observation):
    for prohibition in self.prohibitions:
        if not prohibition.holds(observation):
          return False
    return True

  def update_observation(self, observation, action, x, y):
    """Updates the observation with requested information."""
    # lua is 1-indexed
    observation['NUM_APPLES_AROUND'] = self.get_apples(observation, x, y)
    observation['CUR_CELL_HAS_APPLE'] = True if observation['SURROUNDINGS'][x][y] == 1 else False
    observation['IS_AT_WATER'] = True if observation['SURROUNDINGS'][x][y] == -1 else False
    self.get_territory(observation, x, y)

    action_name = None
    if action >= 7:
      action_name = self.action_to_beam[action - 7]
    observation = self.get_nowalk_action(observation, action_name)
        
    return observation
  
  def get_nowalk_action(self, observation, action_name):
    for action in self.action_to_beam:
        observation[action] = False 
        if action == action_name:
          observation[action] = True

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
  
  def get_territory(self, observation, x, y):
    own_idx = self._index+1
    property_idx = 0
    property_idx = int(observation['PROPERTY'][x][y])
    if property_idx != own_idx and property_idx != 0:
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = True
      # format: ['STOLEN_RECORDS'][thief_id]
      if observation['STOLEN_RECORDS'][property_idx-1] == 1:
        observation['AGENT_HAS_STOLEN'] = True
      else:
        observation['AGENT_HAS_STOLEN'] = False
    else:
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = False
      observation['AGENT_HAS_STOLEN'] = True # free or own property

  def exceeds_map(self, world_rgb, x, y):
    x_max = world_rgb.shape[1] / 8
    y_max = world_rgb.shape[0] / 8
    if x <= 0 or x >= x_max:
      return True
    if y <= 0 or y >= y_max:
      return True
    return False

  def is_done(self, timestep, plan_length, action):
    """Check whether any of the stop criteria are met."""
    if timestep.last():
      return True
    elif plan_length > self._max_depth:
      return True
    elif self.current_obligation != None:
      if action >= 7:
        action_name = self.action_to_beam[action-7]
        return self.current_obligation.satisfied(action_name)
    elif timestep.reward >= 1.0:
      return True
    return False

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
