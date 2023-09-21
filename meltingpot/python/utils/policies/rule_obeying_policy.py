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

import dm_env

from dataclasses import dataclass, field
from typing import Any

from collections import deque
import random

import numpy as np
import hashlib
import pickle

from meltingpot.python.utils.policies import policy
from meltingpot.python.utils.substrates import shapes
from meltingpot.python.utils.policies.agent_timestep import AgentTimestep
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from meltingpot.python.utils.policies.ast_rules import INT_TO_ROLE, ROLE_TO_INT, ROLE_SPRITE_DICT

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False) 

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by ertain environment rules."""

  DEFAULT_MAX_DEPTH = 20
  DEFAULT_COMPLIANCE_COST = 0.1
  DEFAULT_VIOLATION_COST = 0.4
  DEFAULT_TAU = 0.01
  DEFAULT_N_STEPS = 2
  DEFAULT_GAMMA = 0.9999
  DEFAULT_N_ROLLOUTS = 2
  DEFAULT_OBLIGATION_REWARD = 1

  def __init__(self, 
               env: dm_env.Environment, 
               player_idx: int,
               log_output: bool,
               log_weights: bool,
               look: shapes,
               role: str = "free",
               prohibitions: list = DEFAULT_PROHIBITIONS, 
               obligations: list = DEFAULT_OBLIGATIONS,
               max_depth: int = DEFAULT_MAX_DEPTH,
               compliance_cost: float = DEFAULT_COMPLIANCE_COST,
               violation_cost: float = DEFAULT_VIOLATION_COST,
               tau: float = DEFAULT_TAU, 
               n_steps: int = DEFAULT_N_STEPS, 
               gamma: float = DEFAULT_GAMMA,
               n_rollouts: int = DEFAULT_N_ROLLOUTS,
               obligation_reward: int = DEFAULT_OBLIGATION_REWARD) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """

    # CALLING PARAMETER
    self.py_index = player_idx
    self.lua_index = player_idx+1
    self.role = role
    self.look = look
    self.log_output = log_output
    self.log_weights = log_weights
    self.action_spec = env.action_spec()[0]
    self.prohibitions = prohibitions
    self.obligations = obligations

    # CONSTANTS
    self.max_depth = max_depth
    self.compliance_cost = compliance_cost
    self.violation_cost = violation_cost
    self.tau = tau
    # self.action_cost = 1
    # self.epsilon = 0.2
    # self.regrowth_rate = 0.5
    self.n_steps = n_steps
    self.gamma = gamma
    self.n_rollouts = n_rollouts
    self.obligation_reward = obligation_reward
    self.punish_cost = punish_cost
    
    # GLOBAL INITILIZATIONS
    self.history = deque(maxlen=10)
    self.payees = []
    self.riots = []
    self.q_value_log = {}
    self.hash_count = {}
    self.pos_all_apples = []
    goals = ['apple', 'clean', 'pay', 'zap']
    self.V = {goal: {} for goal in goals}
    self.ts_start = None
    self.goal = None
    self.x_max = 15
    self.y_max = 15
    self.self.avg_steps_to_punishment = 3

    # non-physical info
    self.last_zapped = 0
    self.last_paid = 0
    self.last_cleaned = 0
    self.old_pos = None

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
    
    self.relevant_keys = {
      'full': [
            'AGENT_ATE',
            'AGENT_CLAIMED',
            'AGENT_CLEANED',
            'AGENT_PAID',
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_CLEANED',
            'SINCE_AGENT_LAST_PAID',
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
            # 'WATER_LOCATION', # maybe take out again
            'POSITION_OTHERS',
          ],
      'apple': [
            'AGENT_ATE',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SURROUNDINGS',
          ],
        'clean': [
            'AGENT_CLEANED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_CLEANED',
            'SURROUNDINGS',
            #'POSITION_OTHERS',
          ],
          'pay': [
            'AGENT_PAID',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_PAID',
            'SURROUNDINGS',
            'POSITION_OTHERS',
          ],
        'zap': [
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
            'POSITION_OTHERS',
          ],
      }
        
  def step(self, 
           timestep: dm_env.TimeStep,
           actions: list
           ) -> list:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """

      self.x_max = len(timestep.observation['WORLD.RGB'][1]) / 8
      self.y_max = len(timestep.observation['WORLD.RGB'][0]) / 8

      if timestep == timestep.first():
        self.pos_all_apples = list(zip(*np.where(timestep.observation['SURROUNDINGS']== -1)))

      ts_cur = self.add_non_physical_info(timestep=timestep, actions=actions, idx=self.py_index)
      self.ts_start = ts_cur

      # Check if any of the obligations are active
      for obligation in self.obligations:
         if obligation.holds_in_history(self.history):
           self.current_obligation = obligation
           break
         
      self.set_goal()
      ts_cur.goal = self.goal
               
      if self.log_output:
        print(f"player: {self.lua_index} obligation active?: {self.current_obligation != None}")

      if not self.has_policy(ts_cur):
        self.rtdp(ts_cur)

      self.last_inventory = ts_cur.observation['INVENTORY']
      
      return self.get_act(ts_cur, self.py_index)
  
  def set_goal(self) -> None:
    if len(self.current_obligations) != 0:
      self.goal = self.get_cur_obl()
    else:
      self.goal = 'apple'
  
  def has_policy(self, ts_cur: AgentTimestep) -> bool:
    s_next = self.hash_ts(ts_cur)
    if s_next in self.V[self.goal].keys():
      return True
    return False
  
  def surface_feature_to_int(self, surface_feature: str):
      for role_str, _ in ROLE_SPRITE_DICT.items():
          sprite_str = ''.join(ROLE_SPRITE_DICT[role_str]).encode('utf-8')
          if surface_feature == sprite_str:  # Assuming the list contains unique items
              return ROLE_TO_INT[role_str]
      return None  # Return None if the surface feature is not found
  
  def add_non_physical_info(self, timestep: dm_env.TimeStep, actions: list, idx: int) -> AgentTimestep:
    ts = AgentTimestep()
    ts.step_type = timestep.step_type
    dict_observation = self.custom_deepcopy(timestep.observation[idx])
    for obs_name, obs_val in dict_observation.items():
      ts.add_obs(obs_name=obs_name, obs_val=obs_val)

    # not sure whether to subtract 1 or not
    ts.observation['POSITION'][0] = ts.observation['POSITION'][0]-1
    ts.observation['POSITION'][1] = ts.observation['POSITION'][1]-1
    ts.observation['PY_INDEX'] = idx
    ts.observation['AGENT_LOOK'] = self.surface_feature_to_int(ts.observation['AGENT_LOOK'])
    new_pos = ts.observation['POSITION']
    self.update_last_actions(ts.observation, actions[idx])
    ts.observation = self.update_obs_without_coordinates(ts.observation)
    if not self.is_water(ts.observation, new_pos):
      self.update_surroundings(new_pos, ts.observation, idx)
    ts.observation['RIOTS'] = self.update_riots(actions, ts.observation)
    self.set_interpolation_and_dirt_fraction(ts.observation)
    ts.observation['DIRT_FRACTION'] = self.dirt_fraction
    ts.age = self.age
    ts.MAX_LIFE_SPAN = self.MAX_LIFE_SPAN

    return ts
  
  def set_interpolation_and_dirt_fraction(self, obs: dict) -> float:
    dirt_count = np.sum(obs['SURROUNDINGS'] == -3)
    clean_count = np.sum(obs['SURROUNDINGS'] == -4)

    self.dirt_fraction = dirt_count / (dirt_count + clean_count)

    depletion = self.threshold_depletion
    restoration = self.threshold_restoration
    self.interpolation = (self.dirt_fraction - depletion) / (restoration - depletion)
  
  def update_riots(self, actions: list, obs: dict) -> None:
    """Updating the list of riots for every action that has been taken by all agents"""
    for i, action in enumerate(actions): # we need to compute it for all agents
      if action == 7:
        player_who_zapped = i
        if len(self.riots) > 0: # stored in the list
          zapped_agent = self.get_zapped_agent(player_who_zapped, obs)
          if not zapped_agent == None:
            if zapped_agent in self.riots:
              self.riots.remove(zapped_agent)
      
    return self.riots


  def get_zapped_agent(self, player_who_zapped: int, obs: dict) -> int:
    """Returns the Lua index of an agent that was zapped"""
    x, y = 0, 0
    if 0 <= player_who_zapped < len(obs['POSITION_OTHERS']):
      x, y = obs['POSITION_OTHERS'][player_who_zapped][0], obs['POSITION_OTHERS'][player_who_zapped][1]
    else:
      print("Invalid player index:", player_who_zapped)

    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if not self.exceeds_map(i, j):
          if obs['SURROUNDINGS'][i][j] > 0:
            if obs['SURROUNDINGS'][i][j] != player_who_zapped:
              if self.is_facing_agent(obs, (i, j)):
                return obs['SURROUNDINGS'][i][j]
    return None
  
  def update_obs_without_coordinates(self, obs: dict) -> dict:
    cur_pos = np.copy(obs['POSITION'])
    x, y = cur_pos[0], cur_pos[1]
    obs['POSITION_OTHERS'] = self.get_others(obs)
    if not self.exceeds_map(x, y):
      return self.update_observation(obs, x, y)
    else: 
      obs['NUM_APPLES_AROUND'] = 0
      obs['CUR_CELL_HAS_APPLE'] = False
      obs['AGENT_HAS_STOLEN'] = False
      obs['CUR_CELL_IS_FOREIGN_PROPERTY'] = False
    return obs

  def update_observation(self, obs, x, y) -> dict:
    """Updates the observation with requested information."""
    new_obs = self.custom_deepcopy(obs)
    new_obs['POSITION'] = np.array((x, y))
    new_obs['NUM_APPLES_AROUND'] = self.get_apples(new_obs, x, y)
    new_obs['CUR_CELL_HAS_APPLE'] = True if new_obs['SURROUNDINGS'][x][y] == -1 else False
    lua_idx = new_obs['PY_INDEX']+1
    new_obs = self.make_territory_observation(new_obs, x, y, lua_idx)
    return new_obs
  
  def update_last_actions(self, obs: dict, action: int) -> None:
    """Updates the "last done x" section"""
    x, y = obs['POSITION']

    # Create a dictionary to store the counters for each action
    last_counters = {
        7: 'last_zapped',
        8: 'last_cleaned',
        11: 'last_paid'
    }

    for counter_name in last_counters.values():
        setattr(self, counter_name, getattr(self, counter_name) + 1)

    if self.payees == None:
      self.last_paid = 0

    # make sure the stuff actually happens!
    if action in last_counters:
      if action == 8:
        if obs['ORIENTATION'] == 0:
          if self.hit_dirt(obs, x, y):
            self.last_cleaned = 0

      elif action == 11:
        if self.role == "farmer": # other roles don't pay
          if self.last_inventory > 0:
              if not self.payees == None:
                for payee in self.payees:
                  if self.is_close_to_agent(obs, payee):
                    self.last_paid = 0

      elif action == 7:
        for riot in self.riots:
          if self.is_close_to_agent(obs, riot):
            self.last_zapped = 0

    # Update the observations with the updated counters
    self.get_bool_action(observation=obs, action=action)
    obs['SINCE_AGENT_LAST_ZAPPED'] = self.last_zapped
    obs['SINCE_AGENT_LAST_CLEANED'] = self.last_cleaned
    obs['SINCE_AGENT_LAST_PAID'] = self.last_paid

  def hit_dirt(self, obs, x, y) -> bool:
    for i in range(x-2, x+3):
      for j in range(y-2, y):
        if not self.exceeds_map(i, j):
          if obs['SURROUNDINGS'][i][j] == -3:
            return True
    return False

  def get_bool_action(self, observation, action) -> None:
    # keep sorted
    observation['AGENT_ATE'] = True if action == 10 else False
    observation['AGENT_CLAIMED'] = True if action == 9 else False
    observation['AGENT_CLEANED'] = True if action == 8 else False
    observation['AGENT_PAID'] = True if action == 11 else False
    observation['AGENT_ZAPPED'] = True if action == 7 else False
  
  def get_others(self, observation: dict) -> list:
    """Returns the indices of all players in a 2D array."""
    surroundings = observation['SURROUNDINGS']
    positive_values_mask = (surroundings > 0)
    positive_values = surroundings[positive_values_mask]
    sorted_indices = np.argsort(positive_values)
    sorted_positive_indices = np.argwhere(positive_values_mask)[sorted_indices]
    return [(index[0], index[1]) for index in sorted_positive_indices]
  
  """def update_and_append_history(self, timestep: dm_env.TimeStep, actions: list) -> None:
    Append current timestep obsetvation to observation history.
    ts_cur = self.add_non_physical_info(timestep, actions, self.py_index)
    self.history.append(ts_cur.observation)"""

  def maybe_collect_apple(self, observation) -> float:
    x, y = observation['POSITION'][0], observation['POSITION'][1]
    reward_map = observation['SURROUNDINGS']
    has_apple = observation['CUR_CELL_HAS_APPLE']
    if self.exceeds_map(x, y):
      return 0, has_apple
    if reward_map[x][y] == -1:
      observation['SURROUNDINGS'][x][y] = 0
      return 1, False
    return 0, has_apple
  
  def update_surroundings(self, new_pos, observation, idx):
      x, y = new_pos[0], new_pos[1]
      if idx+1 in observation['SURROUNDINGS']:
        cur_pos = list(zip(*np.where(observation['SURROUNDINGS'] == idx+1)))
        if not self.exceeds_map(x, y):
          observation['SURROUNDINGS'][cur_pos[0][0]][cur_pos[0][1]] = 0
          observation['SURROUNDINGS'][x][y] = idx+1

  def increase_action_steps(self, observation: dict) -> None:
      observation['SINCE_AGENT_LAST_ZAPPED'] = observation['SINCE_AGENT_LAST_ZAPPED'] + 1
      observation['SINCE_AGENT_LAST_CLEANED'] = observation['SINCE_AGENT_LAST_CLEANED'] + 1 
      observation['SINCE_AGENT_LAST_PAID'] = observation['SINCE_AGENT_LAST_PAID'] + 1

  def env_step(self, timestep: AgentTimestep, action: int, idx: int) -> AgentTimestep:
      # 1. Unpack observations from timestep
      # TODO: this can be made faster, I believe
      observation = self.custom_deepcopy(timestep.observation)
      observation['PY_INDEX'] = idx
      observation = self.update_obs_without_coordinates(observation)
      self.increase_action_steps(observation)
      self.get_bool_action(observation=observation, action=action)
      next_timestep = AgentTimestep()
      orientation = observation['ORIENTATION']
      cur_inventory = observation['INVENTORY']
      cur_pos = observation['POSITION']
      reward = 0
      action_name = None
      observation['RIOTS'] = self.riots
      observation['COLLECTED_APPLE'] = False

      # 2. Simulate changes to observation based on action
      if action <= 4: # MOVE ACTIONS
        if action == 0 and self.role == 'cleaner':
          # make the cleaner wait for it's paying farmer
          observation['TIME_TO_GET_PAID'] = 0
        new_pos = cur_pos + self.action_to_pos[orientation][action]
        if self.is_water(observation, new_pos):
          new_pos = cur_pos # don't move to water
        observation['POSITION'] = new_pos
        observation['CUR_CELL_HAS_APPLE'] = True if observation['SURROUNDINGS'][new_pos[0]][new_pos[1]] == -1 else False
        new_inventory, has_apple = self.maybe_collect_apple(observation)
        observation['COLLECTED_APPLE'] = True if new_inventory == 1 else False
        observation['CUR_CELL_HAS_APPLE'] = has_apple
        cur_inventory += new_inventory
        self.update_surroundings(new_pos, observation, idx)

      elif action <= 6: # TURN ACTIONS
        observation['ORIENTATION'] = self.action_to_orientation[orientation][action-5]
        
      else: # BEAMS, EAT, & PAY ACTIONS
        cur_pos = tuple(cur_pos)
        x, y = cur_pos[0], cur_pos[1]
        action_name = self.action_to_name[action-7]

        if action_name == "ZAP_ACTION":
          zap_time, riots = self.compute_zap(observation, x, y)
          observation['SINCE_AGENT_LAST_ZAPPED'] = zap_time
          observation['AGENT_ZAPPED'] = True
          observation['RIOTS'] = riots

        if action_name == 'CLAIM_ACTION':
          observation['AGENT_CLAIMED'] = True

        if action_name == 'CLEAN_ACTION':
          last_cleaned_time, num_cleaners = self.compute_clean(observation, x, y)
          observation['SINCE_AGENT_LAST_CLEANED'] = last_cleaned_time
          observation['AGENT_CLEANED'] = True
          observation['TOTAL_NUM_CLEANERS'] = num_cleaners

        if cur_inventory > 0: # EAT AND PAY
          reward, cur_inventory, paid_time = self.compute_eat_pay(action, action_name, 
                                                    cur_inventory, observation)
          observation['SINCE_AGENT_LAST_PAID'] = paid_time
          if action == 11:
            observation['AGENT_PAID'] = True
          else:
            observation['AGENT_ATE'] = True

      observation['INVENTORY'] = cur_inventory
      # observation['WATER_LOCATION'] = list(zip(*np.where(observation['SURROUNDINGS'] <= -3)))

      next_timestep.step_type = dm_env.StepType.MID
      next_timestep.reward = reward
      next_timestep.observation = observation
      next_timestep.goal = self.all_bots[idx].goal
      next_timestep.age = timestep.age + 1
      next_timestep.MAX_LIFE_SPAN = timestep.MAX_LIFE_SPAN

      return next_timestep
  
  def compute_zap(self, observation, x, y):
    last_zapped = observation['SINCE_AGENT_LAST_ZAPPED']
    riots = observation['RIOTS']
    if not self.exceeds_map(x, y):
      for riot in riots:
        if self.is_close_to_agent(observation, riot):
          last_zapped = 0
          riots.remove(riot)

    return last_zapped, riots

  def compute_clean(self, observation, x, y):
    last_cleaned_time = observation['SINCE_AGENT_LAST_CLEANED'] + 1
    num_cleaners = observation['TOTAL_NUM_CLEANERS']
    if not self.role == 'farmer':
      # if facing north and is at water
      if not self.exceeds_map(x, y):
        #if not self.is_water_in_front(observation, x, y):
        if observation['ORIENTATION'] == 0:
          if self.hit_dirt(observation, x, y):
            last_cleaned_time = 0
            num_cleaners = 1

    return last_cleaned_time, num_cleaners
  
  def is_water_in_front(self, observation, x, y):
    orientation = observation['ORIENTATION']
    if orientation == 0:  # North
        return self.is_water(observation, (x, y-1))
    elif orientation == 1:  # East
        return self.is_water(observation, (x+1, y))
    elif orientation == 2:  # South
        return self.is_water(observation, (x, y+1))
    elif orientation == 3:  # West
        return self.is_water(observation, (x-1, y))
    return False
  
  def compute_eat_pay(self, action, action_name, 
                                cur_inventory, observation):

    paid_time = observation['SINCE_AGENT_LAST_PAID']
    reward = 0
    if action >= 10: # eat and pay
      if action_name == "EAT_ACTION":
        reward = 1
        cur_inventory -= 1 # eat
      if action_name == "PAY_ACTION":
        if self.role == "farmer": # other roles don't pay
          if not self.payees == None:
            for payee in self.payees:
              if self.is_close_to_agent(observation, payee):
                cur_inventory -= 1 # pay
                paid_time = 0

    return reward, cur_inventory, paid_time

  """def get_payees(self, observation):
    payees = []
    if isinstance(observation['ALWAYS_PAYING_TO'], np.int32):
      payees.append(observation['ALWAYS_PAYING_TO'])
    else:
      for i in range(len(observation['ALWAYS_PAYING_TO'])):
        if observation['ALWAYS_PAYING_TO'][i] != 0:
          payees.append(i)
    return payees"""
  
  def is_close_to_agent(self, observation, payee):
    x_start = observation['POSITION'][0]-2
    y_start = observation['POSITION'][1]-2
    x_stop = observation['POSITION'][0]+3
    y_stop = observation['POSITION'][1]+3

    for i in range(x_start, x_stop):
      for j in range(y_start, y_stop):
        if not self.exceeds_map(i, j):
          if observation['SURROUNDINGS'][i][j] == payee:
            return self.is_facing_agent(observation, (i, j))
    return False
  
  def is_facing_agent(self, observation, payee_pos):
    orientation = observation['ORIENTATION']
    own_pos = observation['POSITION']

    if payee_pos[0] >= own_pos[0] and orientation == 1:
      return True
    if payee_pos[1] >= own_pos[1] and orientation == 2:
      return True
    if own_pos[0] >= payee_pos[0] and orientation == 3:
      return True
    if own_pos[1] >= payee_pos[1] and orientation == 0:
      return True

    return False
  
  def custom_deepcopy(self, old_obs):
    """Own copy implementation for time efficiency."""
    new_obs = {}
    for key, value in old_obs.items():
        if isinstance(value, np.ndarray):
            new_obs[key] = value.copy() if value.shape else value.item()
        else:
            new_obs[key] = value
    return new_obs
  
  def get_cur_obl(self) -> str:
    """
    Returns the a string with the goal of the obligation.
    """
    if "CLEAN" in self.current_obligations[0].goal:
      return "clean"
    if "PAID" in self.current_obligations[0].goal:
      return "pay"
    if "RIOTS" in self.current_obligations[0].goal:
      return "zap"
    
    return None

  def get_ts_hash_key(self, obs: dict, reward: float, goal: str) -> str:
    try:
        relevant_keys = self.relevant_keys[goal]
    except KeyError:
        print(f"Key {goal} not found in self.relevant_keys")
    #relevant_keys = self.relevant_keys[goal] # define keys
    items = list(obs[key] for key in sorted(obs.keys()) if key in sorted(relevant_keys)) # extract
    #sorted_items = sorted(items, key=lambda x: x[0])

    list_bytes = pickle.dumps(items + [reward]) # make byte arrays
    hash_key = hashlib.sha256(list_bytes).hexdigest()  # hash

    return hash_key

  def hash_ts(self, ts: AgentTimestep):
    """Computes hash for the given timestep observation."""
    return self.get_ts_hash_key(ts.observation, ts.reward, ts.goal)
  
  # from https://github.com/JuliaPlanners/SymbolicPlanners.jl/blob/master/src/planners/rtdp.jl
  def rtdp(self, ts_start: AgentTimestep) -> None:
    # Perform greedy value iteration
    visited = []
    for _ in range(self.n_rollouts):
      ts_cur = ts_start.copy()
      for _ in range(self.max_depth):
        visited.append(ts_cur)
        # greedy rollout giving the next best action
        next_act = self.update(ts_cur)
        # taking nest best action^
        ts_cur = self.env_step(ts_cur, next_act, self.py_index)

    # post-rollout update
    while len(visited) > 0:
      ts_cur = visited.pop()
      _ = self.update(ts_cur)

    return
  
  def get_act(self, ts_cur: AgentTimestep, idx: int, temp=None) -> int:
    hash = self.hash_ts(ts_cur)
    goal = ts_cur.goal
    v_func = self.all_bots[idx].V[goal]

    if hash in v_func.keys():
      if self.log_output:
        print()
        print(f'ROLE: {self.role}')
        print(f'position: {ts_cur.observation["POSITION"]}, orientation: {ts_cur.observation["ORIENTATION"]}, key: {hash}')
        print(v_func[hash])

      next_act = self.get_boltzmann_act(v_func[hash], temp=temp)
      return next_act

  def update(self, ts_cur: AgentTimestep) -> int:
    """Updates state-action pair value function 
    and returns the best action based on that."""
    size = self.action_spec.num_values 
    Q = np.full(size, -1.0)
    Q_wo_rule = np.full(size, -1.0)

    # TODO: change to available_action_history()
    available = self.available_actions(ts_cur.observation)
    s_cur = self.hash_ts(ts_cur)

    if self.log_weights:
      print(f'NEW UPDATE FOR STATE {s_cur}')
    
    for act in range(self.action_spec.num_values):
      ts_next = self.env_step(ts_cur, act, self.py_index)
      s_next = self.init_process_next_ts(ts_next, self.py_index)

      Q[act], Q_wo_rule[act], _  = self.get_bellmann_update(ts_next, s_next, act, available, ts_cur, self.py_index)

    self.V[self.goal][s_cur] = Q
    self.V_wo_rules[s_cur] = Q_wo_rule

    if s_cur not in self.hash_count:
      self.hash_count[s_cur] = 1
      self.q_value_log[s_cur] = [max(Q)]
    else:
      self.hash_count[s_cur] += 1
      self.q_value_log[s_cur].append(max(Q))

    next_act = self.get_boltzmann_act(Q)
    if self.log_weights:
      print('SUUMMARY:')
      print(f"{Q}")
      print(f'next action: {next_act}')
    
    return next_act
  
  def init_process_next_ts(self, ts_cur: AgentTimestep, player_idx: int) -> str:
    pos = ts_cur.observation['POSITION']
    bot = self.all_bots[player_idx]
    goal = ts_cur.goal

    s_next = self.hash_ts(ts_cur)
    # initialize best optimistic guess for next state
    if s_next not in bot.V[goal].keys():
      Q, Q_no_rules = bot.init_heuristic(ts_cur, player_idx)
      bot.V[goal][s_next] = Q
      bot.V_wo_rules[s_next] = Q_no_rules

    return s_next
  
  def is_agent_in_position(self, observation: dict, pos) -> bool:
    surroundings = observation['SURROUNDINGS']
    return surroundings[pos[0], pos[1]] > 0 and surroundings[pos[0], pos[1]] != self.lua_index
  
  def init_heuristic(self, timestep: AgentTimestep, player_idx: int) -> np.array:
    size = self.action_spec.num_values 
    Q = np.full(size, -np.inf)
    Q_no_rules = np.full(size, -np.inf)

    bot = self.all_bots[player_idx]
    goal = timestep.goal
    available = self.available_actions(timestep.observation)

    if self.log_weights:
      print()
      print(f"{timestep.observation['POSITION']} for {self.hash_ts(timestep)}")

    for act in range(size):
      ts_next = bot.env_step(timestep, act, bot.py_index)
      observation = ts_next.observation
      pos = observation['POSITION']
      r_apple = ts_next.reward
      r_obl = 0.0

      norm_cost = 0 if act in available else self.intrinsic_violation_cost
 
      if self.exceeds_map(pos[0], pos[1]):
        continue

      if ts_next.age == self.MAX_LIFE_SPAN:
        continue

      n_steps_to_live = self.MAX_LIFE_SPAN - ts_next.age
      cumulative_action_cost = 0.0
      for i in range(n_steps_to_live):
          cumulative_action_cost += self.default_action_cost * self.gamma**(i)
    
      pos_cur_apples = bot.get_cur_obj_pos(observation['SURROUNDINGS'], object_idx = -1)
      r_inv_apple = self.apple_reward * observation['INVENTORY']
      r_cur_apples = bot.get_discounted_reward(pos_cur_apples, pos, ts_next.age, goal='apple')

      r_apple += r_cur_apples + r_inv_apple

      if self.log_weights:
        print(f"len cur_apples: {len(pos_cur_apples)}, reward: {ts_next.reward}, r_cur_apples: {r_cur_apples}, r_inv_apple: {r_inv_apple}")

      if goal != 'apple':
        pos_cur_obl = bot.get_cur_obl_pos(observation)
        r_cur_obl = bot.get_discounted_reward(pos_cur_obl, pos, ts_next.age, goal)
        r_fulfilled_obl = self.obligation_reward if bot.current_obligations[0].satisfied(observation) else 0

        r_obl = r_cur_obl + r_fulfilled_obl - self.default_action_cost
        
        if self.log_weights:
          # print(f"len pos_fut_obl: {len(pos_fut_obl)}, reward: {r_fut_obl}, fulfilled: {r_fulfilled_obl}")
          print(f"len pos_cur_obl: {len(pos_cur_obl)}, reward: {r_cur_obl}, fulfilled: {r_fulfilled_obl}")

      Q[act] = r_apple - norm_cost - cumulative_action_cost if goal == 'apple' else r_obl - norm_cost - cumulative_action_cost
      Q_no_rules[act] = r_apple - cumulative_action_cost

    if self.log_weights:
      print(Q)

    return Q, Q_no_rules

  def get_discounted_reward(self, target_pos, own_pos, age, goal) -> float:
    reward = 0.0
    r_amount = self.apple_reward if goal == 'apple' else self.obligation_reward

    n_steps_to_live = self.MAX_LIFE_SPAN - age
    cumulative_action_cost = 0.0

    for pos in target_pos:
      n_steps_to_reward = int(self.manhattan_dis(pos, own_pos)) + 1

      if goal != 'apple': # Handle action costs
        n_steps_left = min(n_steps_to_reward, n_steps_to_live)
        for i in range(n_steps_left):
           cumulative_action_cost += self.default_action_cost * self.gamma**(i)
        reward -= cumulative_action_cost

      if n_steps_to_reward > self.MAX_LIFE_SPAN - age:
        continue

      reward += r_amount * self.gamma**(n_steps_to_reward) # Positive reward for eating apple

    return reward
  
  def find_closest_dirt(self, grid, pos):
    closest_distance = float('inf')
    closest_dirt = None
    this_pos = tuple(pos)
    # start_row, start_col = pos[0], pos[1]
    
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == self.dirt_index:  # if cell contains dirt
                possible_dirt = tuple((row_index, col_index))
                distance = self.manhattan_dis(this_pos, possible_dirt)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_dirt = [tuple((row_index, col_index))]
    
    return closest_dirt
  
  def get_cur_obl_pos(self, observation: dict) -> list:
    if self.goal == 'clean':
      pos = observation["POSITION"]
      surroundings = observation["SURROUNDINGS"]
      return self.find_closest_dirt(surroundings, pos)
    else:
      return self.get_agent_list(observation)
        
  def get_agent_list(self, observation: dict) -> list:
    agent_idx_vec = self.payees if self.goal == "pay" and observation['INVENTORY'] != 0 else []

    if self.goal == "zap":
      agent_idx_vec = self.riots
    
    agents_pos = []
    for agent in agent_idx_vec:
      if not agent == 0: # len(agent_idx) == num_agents, one-hot-encoded for payees
        agents_pos += list(zip(*np.where(observation['SURROUNDINGS'] == agent)))

    return agents_pos
  
  def get_dirt_conditioned_regrowth_rate(self) -> float:
    interpolation = min(self.interpolation, 1.0)
    probability = self.max_apple_growth_rate * interpolation
    return probability

  def manhattan_dis(self, pos_cur, pos_goal) -> int:
    try:
        distance = abs(pos_cur[0] - pos_goal[0]) + abs(pos_cur[1] - pos_goal[1])
    except TypeError:
        print(f"TypeError occurred! pos_cur={pos_cur}, type(pos_cur)={type(pos_cur)}, pos_goal={pos_goal}, type(pos_goal)={type(pos_goal)}")
        raise  # re-raise the exception after printing debug info
    return distance
  
  def in_unreachable_water(self, obs: AgentTimestep) -> list:
    unreachable = []
    surroundings = obs['SURROUNDINGS']
    for i in range(len(surroundings)):
      for j in range(len(surroundings[0])):
        if surroundings[i][j] <= -2:
          if not surroundings[i][j+3] == 0:
            unreachable.append(tuple((i, j)))

    return unreachable

  def get_cur_obj_pos(self, surroundings: np.array, object_idx: int) -> list:
    return list(zip(*np.where(surroundings== object_idx)))
  
  def compute_boltzmann(self, q_values: list, tau=None):

    if tau == None:
      tau = self.tau

    mean_q_value = np.mean(q_values)
    transform_q_values = (q_values - mean_q_value) / np.std(q_values)
    probs = np.exp(transform_q_values / tau)
    probs /= probs.sum() # normalized
    
    # Check and handle NaN values
    if np.any(np.isnan(probs)):
        #print("Warning: NaN values detected in probabilities. Using uniform distribution.")
        probs = np.ones_like(q_values) / len(q_values)

    return probs
  
  def get_boltzmann_act(self, q_values: list, temp=None) -> int:

    if temp == 0:
        return np.argmax(q_values)
    
    probs = self.compute_boltzmann(q_values)
    action = np.random.choice(len(q_values), p=probs) 
    return action

  def get_bellmann_update(self, ts_next: AgentTimestep, s_next: str, act: int, available: list, ts_cur: AgentTimestep, player_idx: int) -> float:
    observation = ts_next.observation
    pos = observation['POSITION']
    bot = self.all_bots[player_idx]
    goal = ts_next.goal

    r_forward = max(bot.V[goal][s_next]) * self.gamma
    r_cur = ts_next.reward - self.default_action_cost

    r_forward_no_rule = max(bot.V_wo_rules[s_next]) * self.gamma
    r_no_rule = ts_next.reward - self.default_action_cost

    if len(bot.current_obligations) != 0:        
      r_cur = 0
      if bot.current_obligations[0].satisfied(observation):
        r_cur = bot.obligation_reward

    norm_cost = 0 if act in available else self.intrinsic_violation_cost # rule violation
    if bot.riot_rule_is_active():
      norm_cost += self.punish_cost * self.gamma**self.avg_steps_to_punishment

    if bot.is_agent_in_position(ts_cur.observation, pos):
      r_cur -= self.element_blocking_cost
      r_no_rule -= self.element_blocking_cost

    if bot.is_water(ts_cur.observation, pos):
      r_cur -= self.element_blocking_cost * 10
      r_no_rule -= self.element_blocking_cost * 10

    if self.log_weights:
      print()
      print(f'{ts_cur.observation["POSITION"]} for {act} to {ts_next.observation["POSITION"]} gives\t{r_forward} + {r_cur} - {r_cur}; {s_next}')

    v_rules = r_forward + r_cur - norm_cost
    v_wo_rule = r_forward_no_rule + r_no_rule
    v_one_ts_rule = v_wo_rule - norm_cost

    return v_rules, v_wo_rule, v_one_ts_rule
  
  """def a_star(self, s_start: int) -> list[int]:
    # Perform a A* search to generate plan.
    PRIO_QUEUE, S_ORDERED, action = EnumPriorityQueue(), [], 0
    PRIO_QUEUE.put(0, s_start) # ordered by reward
    self.h_vals[s_start] = self.heuristic(s_start)
    came_from = {s_start: s_start}
    g_table = {s_start: 0}
    for s_goal in self.s_goal: # g_vals for current goals
      g_table[s_goal] = float('inf')
    depth = 0

    while not PRIO_QUEUE.empty():
      depth += 1
      s_cur = PRIO_QUEUE.get()
      S_ORDERED.append(s_cur)
      cur_timestep, _ = self.unhash(s_cur)

      if self.is_goal(s_cur):
        OPEN = True
        S_ORDERED = self.reconstruct_path(s_cur)
        return OPEN, S_ORDERED, [], came_from
      
      if depth >= self.max_depth:
        return PRIO_QUEUE, S_ORDERED, g_table, came_from

      for action in range(self.action_spec.num_values):
        # simulate environment for that action
        next_timestep = self.env_step(cur_timestep, action)
        s_next = self.hash_ts_and_action(cur_timestep, action)

        if s_next not in S_ORDERED: # has not been visited yet
          new_cost = g_table[s_cur] + self.action_cost(s_cur, s_next, action)
          if s_next not in g_table:
            g_table[s_next] = float('inf')
          if new_cost < g_table[s_next]:  # conditions for updating cost
            g_table[s_next] = new_cost
            came_from[s_next] = s_cur

          self.h_vals[s_next] = self.heuristic(s_next)
          priority = g_table[s_next] + self.h_vals[s_next]
          PRIO_QUEUE.put(priority=priority, item=s_next)

      if depth >= self.max_depth:
        break

    return PRIO_QUEUE, S_ORDERED, g_table, came_from"""
  
  def intersection(self, lst1, lst2):
    out = [value for value in lst1 if value in lst2]
    return out
  
  def riot_rule_is_active(self):
    for rule in self.obligations:
      if "RIOTS" in rule.make_str_repr():
        return True
    return False
  
  def available_action_history(self):
    action_list = self.available_actions(self.history[0])
    for obs in list(self.history)[1:]:
      new_list = self.available_actions(obs)
      action_list = self.intersection(new_list, action_list)
      
    return action_list

  def available_actions(self, obs) -> list[int]:
    """Return the available actions at a given timestep."""
    actions = []
    observation = self.custom_deepcopy(obs)
    cur_pos = np.copy(observation['POSITION'])

    for action in range(self.action_spec.num_values):
      x, y = self.update_coordinates_by_action(action, 
                                              cur_pos,
                                              observation)

      if self.exceeds_map(x, y):
        continue

      new_obs = self.update_observation(observation, x, y)
      action_name = self.get_action_name(action)
      
      if self.check_all(new_obs, action_name):
          actions.append(action)

    return actions
  
  def update_coordinates_by_action(self, action, cur_pos, observation):
    x, y = cur_pos[0], cur_pos[1]
    orientation = observation['ORIENTATION']
    if action <= 4: # move actions
      new_pos = cur_pos + self.action_to_pos[orientation][action]
      x, y = new_pos[0], new_pos[1]

    return x, y
  
  def check_all(self, observation, action):
    for prohibition in self.prohibitions:
       if prohibition.holds(observation, action):
          return False
    return True
  
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
    surroundings = observation['SURROUNDINGS']
    x_min, x_max = max(0, x - 1), min(len(surroundings), x + 2)
    y_min, y_max = max(0, y - 1), min(len(surroundings[0]), y + 2)

    subarray = surroundings[x_min:x_max, y_min:y_max]
    apple_count = np.sum(subarray == -1)

    # Since the center cell is included, remove it if it's an apple.
    if surroundings[x, y] == -1:
        apple_count -= 1

    return apple_count
    
  def make_territory_observation(self, observation, x, y, lua_idx):
    """
    Adds values for territory components to the observation dict.
      AGENT_HAS_STOLEN: if the owner of the current cell has stolen
          from the current agent. True for own and free property, too.
      CUR_CELL_IS_FOREIGN_PROPERTY: True if current cell does not
          belong to current agent.
    """
    new_obs = self.custom_deepcopy(observation)
    new_obs['POSITION'] = np.array((x, y))
    property_idx = int(new_obs['PROPERTY'][x][y])
    new_obs['AGENT_HAS_STOLEN'] = True

    if property_idx == 0 or property_idx == lua_idx:
      new_obs['CUR_CELL_IS_FOREIGN_PROPERTY'] = False

    else:
      new_obs['CUR_CELL_IS_FOREIGN_PROPERTY'] = True
      if new_obs['STOLEN_RECORDS'][lua_idx-1] != 1:
        new_obs['AGENT_HAS_STOLEN'] = False

    return new_obs

  def is_water(self, observation, pos):
    x, y = pos[0], pos[1]
    if self.exceeds_map(x, y):
      return True
    if observation["SURROUNDINGS"][x][y] <= -2:
      return True
    return False

  def exceeds_map(self, x, y):
    """Returns True if current cell index exceeds game map."""
    if x < 0 or x >= self.x_max-1:
      return True
    if y < 0 or y >= self.y_max-1:
      return True
    return False

  def is_done(self, timestep: AgentTimestep):
    """Check whether any of the break criteria are met."""
    if timestep.last():
      return True
    elif len(self.current_obligations) != 0:
      return self.current_obligations[0].satisfied(
        timestep.observation)
    elif timestep.reward >= 1.0:
      return True
    return False

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
