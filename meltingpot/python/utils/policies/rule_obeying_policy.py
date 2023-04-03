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

from collections import deque

from ast import parse

from pysmt.shortcuts import *

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule

import numpy as np
from copy import deepcopy

from meltingpot.python.utils.policies import policy

# VARIABLES
foreign_property = parse("lambda obs : obs['CUR_CELL_IS_FOREIGN_PROPERTY']")
cur_cell_has_apple = parse("lambda obs : obs['CUR_CELL_HAS_APPLE']")
num_apples_around = parse("lambda obs : obs['NUM_APPLES_AROUND']")
agent_has_stolen = parse("lambda obs : obs['AGENT_HAS_STOLEN']")
num_cleaners = parse("lambda obs : obs['TOTAL_NUM_CLEANERS']")
sent_last_payment = parse("lambda obs : obs['SINCE_AGENT_LAST_PAYED']")
did_last_cleaning = parse("lambda obs : obs['SINCE_AGENT_LAST_CLEANED']")
received_last_payment = parse("lambda obs : obs['SINCE_RECEIVED_LAST_PAYMENT']")

# PRECONDITIONS AND GOALS FOR OBLIGATIONS
cleaning_precondition_free = parse("lambda obs : num_cleaners(obs) < 1")
cleaning_goal_free = parse("lambda obs : num_cleaners(obs) >= 1")
payment_precondition_farmer = parse("lambda obs : sent_last_payment(obs) > 1")
payment_goal_farmer = parse("lambda obs : sent_last_payment(obs) <= 1")
cleaning_precondition_cleaner = parse("lambda obs : did_last_cleaning(obs) > 1")
cleaning_goal_cleaner = parse("lambda obs : did_last_cleaning(obs) <= 1")
payment_precondition_cleaner = parse("lambda obs : received_last_payment(obs) > 1")
payment_goal_cleaner = parse("lambda obs : received_last_payment(obs) <= 1")

DEFAULT_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precondition_free, cleaning_goal_free),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precondition_farmer, payment_goal_farmer, "farmer"),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precondition_cleaner, cleaning_goal_cleaner, "cleaner"),
  # if you're a cleaner, wait until you've received a payment
  ObligationRule(payment_precondition_cleaner, payment_goal_cleaner, "cleaner")
]

# PRECONDITIONS FOR PROHIBTIONS
harvest_apple_precondition = parse("lambda obs : cur_cell_has_apple(obs) \
                                   and num_apples_around(obs) < 3")
steal_from_forgein_cell_precondition = parse("lambda obs : cur_cell_has_apple(obs) \
                                   and not agent_has_stolen(obs)")
  
DEFAULT_PROHIBITIONS = [
  # don't go if <2 apples around
  ProhibitionRule(harvest_apple_precondition, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precondition, 'MOVE_ACTION'),
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
               role: str = "free",
               prohibitions: list = DEFAULT_PROHIBITIONS, 
               obligations: list = DEFAULT_OBLIGATIONS) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """
    self._index = player_idx
    self.role = role
    self._max_depth = 30
    self.action_spec = env.action_spec()[0]
    self.prohibitions = prohibitions
    self.obligations = obligations
    self.current_obligation = None
    self.history = deque(maxlen=5)
    self.payees = []
    if self.role == 'farmer':
      self.payees = None

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
    # non-move actions
    self.action_to_name = [
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

      self.history.append(timestep.observation)
      # Check if any of the obligations are active
      self.current_obligation = None
      for obligation in self.obligations:
         if obligation.holds_in_history(self.history, self.role):
           self.current_obligation = obligation
           break
         
      print(f"player: {self._index} current_obligation active?: {self.current_obligation != None}")

      # Select an action based on the first satisfying rule
      return self.a_star(timestep)
  
  def maybe_collect_apple(self, observation) -> float:
    # lua is one indexed
    x, y = observation['POSITION'][0]-1, observation['POSITION'][1]-1
    reward_map = observation['SURROUNDINGS']
    if self.exceeds_map(observation['WORLD.RGB'], x, y):
      return 0
    if reward_map[x][y] == -2:
      return 1
    return 0

  def env_step(self, timestep: dm_env.TimeStep, action) -> dm_env.TimeStep:
      # Unpack observations from timestep
      observation = deepcopy(timestep.observation)
      orientation = observation['ORIENTATION'].item()
      reward = timestep.reward
      cur_inventory = observation['INVENTORY']
      action_name = None

      # Simulate changes to observation based on action
      if action <= 4: # move actions
        if action == 0 and self.role == 'cleaner':
          observation['SINCE_RECEIVED_LAST_PAYMENT'] = 0
        observation['POSITION'] += self.action_to_pos[orientation][action]
        cur_inventory += self.maybe_collect_apple(observation)
      
      elif action <= 6: # turn actions
        action = action - 5 # indexing starts at 0
        observation['ORIENTATION'] = np.array(self.action_to_orientation
                                             [orientation][action])
        
      else: # beams, pay, and eat actions
        cur_pos = tuple(observation['POSITION'])
        x, y = cur_pos[0]-1, cur_pos[1]-1 # lua is 1-indexed
        action_name = self.action_to_name[action-7]

        if action_name == 'CLEAN_ACTION':
          if not self.role == 'farmer':
            # if facing north and is at water
            if not self.exceeds_map(observation['WORLD.RGB'], x, y):
              if observation['ORIENTATION'] == 0 \
                and observation['SURROUNDINGS'][x][y] == -1:
                observation['SINCE_AGENT_LAST_CLEANED'] = 0
                observation['TOTAL_NUM_CLEANERS'] = 1

        if cur_inventory > 0:
          if action >= 10: # eat and pay
            if action_name == "EAT_ACTION":
              reward += 1
              cur_inventory -= 1 # eat
            if action_name == "PAY_ACTION":
              if self.role == "farmer":
                if self.payees == None:
                  self.payees = self.get_payees(observation)
                for payee in self.payees:
                  if self.is_close_to_agent(observation, payee):
                    cur_inventory -= 1 # pay
                    observation['SINCE_AGENT_LAST_PAYED'] = 0

      observation['INVENTORY'] = cur_inventory

      return dm_env.TimeStep(step_type=dm_env.StepType.MID,
                                     reward=reward,
                                     discount=1.0,
                                     observation=observation,
                                     )
  def get_payees(self, observation):
    payees = []
    for i in range(len(observation['ALWAYS_PAYING_TO'])):
      if observation['ALWAYS_PAYING_TO'][i] != 0:
        payees.append(observation['ALWAYS_PAYING_TO'][i])
    return payees
  
  def is_close_to_agent(self, observation, payee):
    x_start = observation['POSITION'][0]-1
    y_start = observation['POSITION'][1]-1
    x_stop = observation['POSITION'][0]+1
    y_stop = observation['POSITION'][1]+1

    for i in range(x_start, x_stop):
      for j in range(y_start, y_stop):
        if not self.exceeds_map(observation['WORLD.RGB'], i, j):
          if observation['SURROUNDINGS'][i][j] == payee:
            return self.is_facing_agent(observation, (i, j))
    return False
  
  def is_facing_agent(self, observation, payee_pos):
    orientation = observation['ORIENTATION'].item()
    own_pos = observation['POSITION']

    if payee_pos[0] > own_pos[0] and orientation == 1:
      return True
    if payee_pos[1] > own_pos[1] and orientation == 2:
      return True
    if own_pos[0] > payee_pos[0] and orientation == 3:
      return True
    if own_pos[1] > payee_pos[1] and orientation == 0:
      return True

    return False
      
  def reconstruct_path(self, came_from: dict, coordinates: tuple) -> list:
    path = np.array([coordinates[2]])
    while coordinates in came_from.keys():
      if not coordinates == came_from[coordinates]:
        coordinates = came_from[coordinates]
        if not coordinates[2] == 0:
          path = np.append(path, coordinates[2])
      else:
        break
    path = np.flip(path)
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
      cur_orientation = cur_timestep.observation['ORIENTATION'].item()
      cur_depth = priority_item.priority

      if self.is_done(cur_timestep, cur_depth):
        return self.reconstruct_path(came_from, (cur_position, cur_orientation, cur_action))

      # Get the list of actions that are possible and satisfy the rules
      available_actions = self.available_actions(cur_timestep)

      for action in available_actions:
        # simulate environment for that action
        next_timestep = self.env_step(cur_timestep, action)
        next_position = tuple(next_timestep.observation['POSITION'])
        next_orientation = next_timestep.observation['ORIENTATION'].item()
        # record path if it's new or has higer reward
        if not (next_position, next_orientation, action) in came_from.keys() \
          or next_timestep.reward > cur_timestep.reward:
          came_from[(next_position, next_orientation, action)] = (cur_position,
                                                                  cur_orientation,
                                                                  cur_action)
          # turning twice never gets prioritized  
          new_depth = cur_depth+1
          # don't count depth for double turns 
          if (action == cur_action == 5) or (action == cur_action == 6):
            new_depth = new_depth-1
          queue.put(PrioritizedItem(priority=new_depth,
                                    tie_break=next_timestep.reward*(-1), # ascending
                                    item=(next_timestep, action))
                                    )
    return False

  def available_actions(self, timestep: dm_env.TimeStep) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    observation = deepcopy(timestep.observation)

    for action in range(self.action_spec.num_values):
      cur_pos = deepcopy(observation['POSITION'])
      orientation = observation['ORIENTATION'].item()
      if action <= 4: # move actions
        cur_pos += self.action_to_pos[orientation][action]
      # lua is 1-indexed
      x, y = cur_pos[0]-1, cur_pos[1]-1
      if self.exceeds_map(observation['WORLD.RGB'], x, y):
        continue

      observation = self.update_observation(observation, x, y)
      action_name = self.get_action_name(action)
      if self.check_all(observation, action_name):
        actions.append(action)

    return actions
  
  def check_all(self, observation, action):
    for prohibition in self.prohibitions:
        if prohibition.holds(observation, action):
          return False
    return True

  def update_observation(self, observation, x, y):
    """Updates the observation with requested information."""
    
    observation['NUM_APPLES_AROUND'] = self.get_apples(observation, x, y)
    observation['CUR_CELL_HAS_APPLE'] = TRUE() if \
      observation['SURROUNDINGS'][x][y] == 1 else FALSE()
    self.make_territory_observation(observation, x, y)

    return observation
  
  def get_action_name(self, action):
    """Add bool values for taken action to the observation dict."""
    if action == 0:
      return "NOOP_ACTION"
    elif action <= 4:
      return "MOVE_ACTION"
    elif action <= 6:
      return "TURN_ACTION"
    else:
      return self.action_to_name[action-7]
  
  def get_apples(self, observation, x, y):
    """Returns the sum of apples around a certain position."""
    sum = 0
    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if not self.exceeds_map(observation['WORLD.RGB'], i, j):
          if observation['SURROUNDINGS'][i][j] == -2:
            sum += 1
    
    return Int(sum)
    
  def make_territory_observation(self, observation, x, y):
    """
    Adds values for territory components to the observation dict.
      AGENT_HAS_STOLEN: if the owner of the current cell has stolen
          from the current agent. True for own and free property, too.
      CUR_CELL_IS_FOREIGN_PROPERTY: True if current cell does not
          belong to current agent.
    """
    own_idx = self._index+1
    property_idx = int(observation['PROPERTY'][x][y])
    observation['AGENT_HAS_STOLEN'] = TRUE()

    if property_idx != own_idx and property_idx != 0:
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = True
      if observation['STOLEN_RECORDS'][property_idx-1] != 1:
        observation['AGENT_HAS_STOLEN'] = FALSE()
    else:
      # free or own property
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = FALSE()

  def exceeds_map(self, world_rgb, x, y):
    """Returns True if current cell index exceeds game map."""
    x_max = world_rgb.shape[1] / 8
    y_max = world_rgb.shape[0] / 8
    if x <= 0 or x >= x_max-1:
      return True
    if y <= 0 or y >= y_max-1:
      return True
    return False

  def is_done(self, timestep, plan_length):
    """Check whether any of the break criteria are met."""
    if timestep.last():
      return True
    elif plan_length > self._max_depth:
      return True
    elif self.current_obligation != None:
      return self.current_obligation.satisfied(
        timestep.observation, self.role)
    elif timestep.reward >= 1.0:
      return True
    return False
  
  """def __deepcopy__(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    new_timestep = dm_env.TimeStep()
    orientation = observation['ORIENTATION'].item()
    reward = timestep.reward
    cur_inventory = observation['INVENTORY']
    observation['SINCE_RECEIVED_LAST_PAYMENT']
    observation['POSITION'] += self.action_to_pos[orientation][action]
    observation['SINCE_AGENT_LAST_CLEANED'] = 0
    observation['TOTAL_NUM_CLEANERS'] = 1
    return new_timestep"""

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
