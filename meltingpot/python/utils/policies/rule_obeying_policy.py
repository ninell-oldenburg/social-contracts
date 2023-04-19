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

import numpy as np
from copy import deepcopy

from meltingpot.python.utils.policies import policy

from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    tie_break: float
    item: Any=field(compare=False)
    

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by ertain environment rules."""

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
            # N    # F    # B    # SR   # SL
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
           ) -> list:
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
    x, y = observation['POSITION'][0], observation['POSITION'][1]
    reward_map = observation['SURROUNDINGS']
    if self.exceeds_map(observation['WORLD.RGB'], x, y):
      return 0
    if reward_map[x][y] == -3:
      return 1
    return 0

  def env_step(self, timestep: dm_env.TimeStep, action) -> dm_env.TimeStep:
      # 1. Unpack observations from timestep
      observation = self.deepcopy_dict(timestep.observation)
      orientation = observation['ORIENTATION']
      cur_inventory = observation['INVENTORY']
      cur_pos = observation['POSITION']
      reward = timestep.reward
      action_name = None

      # 2. Simulate changes to observation based on action
      if action <= 4: # MOVE ACTIONS
        if action == 0 and self.role == 'cleaner':
          # make the cleaner wait for it's paying farmer
          observation['TIME_TO_GET_PAYED'] = 0
        observation['POSITION'] = cur_pos + self.action_to_pos[orientation][action]
        cur_inventory += self.maybe_collect_apple(observation)

      elif action <= 6: # TURN ACTIONS
        observation['ORIENTATION'] = self.action_to_orientation[orientation][action-5]
        
      else: # BEAMS, EAT, & PAY ACTIONS
        cur_pos = tuple(cur_pos)
        x, y = cur_pos[0], cur_pos[1]
        action_name = self.action_to_name[action-7]

        if action_name == 'CLEAN_ACTION':
          last_cleaned_time, num_cleaners = self.compute_clean(observation, x, y)
          observation['SINCE_AGENT_LAST_CLEANED'] = last_cleaned_time
          observation['TOTAL_NUM_CLEANERS'] = num_cleaners

        if cur_inventory > 0: # EAT AND PAY
          reward, cur_inventory, payed_time = self.compute_eat_pay(action, action_name, 
                                                    reward, cur_inventory, observation)
          observation['SINCE_AGENT_LAST_PAYED'] = payed_time

      observation['INVENTORY'] = cur_inventory

      return dm_env.TimeStep(step_type=dm_env.StepType.MID,
                                     reward=reward,
                                     discount=1.0,
                                     observation=observation,
                                     )

  def compute_clean(self, observation, x, y):
    last_cleaned_time = observation['SINCE_AGENT_LAST_CLEANED']
    num_cleaners = observation['TOTAL_NUM_CLEANERS']
    if not self.role == 'farmer':
      # if facing north and is at water
      if not self.exceeds_map(observation['WORLD.RGB'], x, y):
        if observation['ORIENTATION'] == 0 \
          and observation['SURROUNDINGS'][x][y] == -1:
          last_cleaned_time = 0
          num_cleaners = 1

    return last_cleaned_time, num_cleaners
  
  def compute_eat_pay(self, action, action_name, reward, 
                                cur_inventory, observation):
    payed_time = observation['SINCE_AGENT_LAST_PAYED']
    if action >= 10: # eat and pay
      if action_name == "EAT_ACTION":
        reward += 1
        cur_inventory -= 1 # eat
      if action_name == "PAY_ACTION":
        if self.role == "farmer": # other roles don't pay
          if self.payees == None:
            self.payees = self.get_payees(observation)
          for payee in self.payees:
            if self.is_close_to_agent(observation, payee):
              cur_inventory -= 1 # pay
              payed_time = 0

    return reward, cur_inventory, payed_time
  
  def deepcopy_dict(self, old_obs):
    new_obs = {}
    for key in old_obs:
      if isinstance(old_obs[key], np.ndarray):
        if old_obs[key].shape == ():
          new_obs[key] = old_obs[key].item()
        elif old_obs[key].shape == (1,):
          new_obs[key] = old_obs[key][0] # unpack numpy array
        else:
          new_obs[key] = np.copy(old_obs[key])
      else:
        new_obs[key] = old_obs[key]

    return new_obs

  def get_payees(self, observation):
    payees = []
    for i in range(len(observation['ALWAYS_PAYING_TO'])):
      if observation['ALWAYS_PAYING_TO'][i] != 0:
        payees.append(observation['ALWAYS_PAYING_TO'][i])
    return payees
  
  def is_close_to_agent(self, observation, payee):
    x_start = observation['POSITION'][0]-1
    y_start = observation['POSITION'][1]-1
    x_stop = observation['POSITION'][0]+2
    y_stop = observation['POSITION'][1]+2

    for i in range(x_start, x_stop):
      for j in range(y_start, y_stop):
        if not self.exceeds_map(observation['WORLD.RGB'], i, j):
          if observation['SURROUNDINGS'][i][j] == payee:
            return self.is_facing_agent(observation, (i, j))
    return False
  
  def is_facing_agent(self, observation, payee_pos):
    orientation = observation['ORIENTATION']
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
        path = np.append(path, coordinates[2])
      else:
        break
    path = np.flip(path)
    return path
  
  def a_star(self, timestep: dm_env.TimeStep) -> list[int]:
    """Perform a A* search to generate plan."""
    queue, action, came_from = PriorityQueue(), 0, {}
    timestep = timestep._replace(reward=0.0)
    observation = timestep.observation
    observation['POSITION'] = np.array([observation['POSITION'][0]-1, 
                              observation['POSITION'][1]-1]) # lua is 1-indexed
    observation['ORIENTATION'] = observation['ORIENTATION'].item()
    queue.put(PrioritizedItem(0, 0, (timestep, action))) # ordered by reward

    while not queue.empty():
      priority_item = queue.get()
      cur_timestep, cur_action = priority_item.item
      cur_pos = tuple(cur_timestep.observation['POSITION'])
      cur_orient = cur_timestep.observation['ORIENTATION']
      cur_depth = priority_item.priority

      if self.is_done(cur_timestep, cur_depth):
        return self.reconstruct_path(came_from, (cur_pos, cur_orient, cur_action))

      # Get the list of actions that are possible and satisfy the rules
      available_actions = self.available_actions(cur_timestep.observation)

      for action in available_actions:
        # simulate environment for that action
        next_timestep = self.env_step(cur_timestep, action)
        next_pos = tuple(next_timestep.observation['POSITION'])
        next_orient = next_timestep.observation['ORIENTATION']
        # record path if it's new or has higer reward
        if not (next_pos, next_orient, action) in came_from.keys() \
          or next_timestep.reward > cur_timestep.reward:
          new_depth = cur_depth+1
          came_from[(next_pos, next_orient, action)] = (cur_pos, cur_orient, cur_action)
          # make 180 degree turns possible (otherwise wouldn't get prioritized)
          if (action == cur_action == 5) or (action == cur_action == 6):
            new_depth = new_depth-1

          queue.put(PrioritizedItem(priority=new_depth,
                                    tie_break=next_timestep.reward*(-1), # ascending
                                    item=(next_timestep, action))
                                    )
    return [0] # return noop action if path finding unsuccessful

  def available_actions(self, obs) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    observation = self.deepcopy_dict(obs)
    cur_pos = np.copy(observation['POSITION'])

    for action in range(self.action_spec.num_values):
      x, y = cur_pos[0], cur_pos[1]
      orientation = observation['ORIENTATION']
      if action <= 4: # move actions
        new_pos = cur_pos + self.action_to_pos[orientation][action]
        x, y = new_pos[0], new_pos[1]

      if self.exceeds_map(observation['WORLD.RGB'], x, y):
        continue

      if observation['SURROUNDINGS'][x][y] == -2: # non-dirt water
        continue

      new_obs = self.update_observation(observation, x, y)
      action_name = self.get_action_name(action)
      if self.check_all(new_obs, action_name):
        actions.append(action)

    return actions
  
  def check_all(self, observation, action):
    for prohibition in self.prohibitions:
       if prohibition.holds(observation, action):
          return False
        
    return True

  def update_observation(self, obs, x, y):
    """Updates the observation with requested information."""
    obs['NUM_APPLES_AROUND'] = self.get_apples(obs, x, y)
    obs['CUR_CELL_HAS_APPLE'] = True if obs['SURROUNDINGS'][x][y] == -3 else False
    self.make_territory_observation(obs, x, y)

    return obs
  
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
          if not (i == x and j == y): # don't count target apple
            if observation['SURROUNDINGS'][i][j] == -3:
              sum += 1
    
    return sum
    
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
    observation['AGENT_HAS_STOLEN'] = True

    if property_idx != own_idx and property_idx != 0:
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = True
      if observation['STOLEN_RECORDS'][property_idx-1] != 1:
        observation['AGENT_HAS_STOLEN'] = False
    else:
      # free or own property
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = False

  def exceeds_map(self, world_rgb, x, y):
    """Returns True if current cell index exceeds game map."""
    x_max = world_rgb.shape[1] / 8
    y_max = world_rgb.shape[0] / 8
    if x < 0 or x >= x_max-1:
      return True
    if y < 0 or y >= y_max-1:
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

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
