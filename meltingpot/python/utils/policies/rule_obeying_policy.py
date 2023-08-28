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

    # non-physical info
    self.last_zapped = 0
    self.last_payed = 0
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
            'AGENT_PAYED',
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_CLEANED',
            'SINCE_AGENT_LAST_PAYED',
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
            'WATER_LOCATION', # maybe take out again
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
            'POSITION_OTHERS',
          ],
          'pay': [
            'AGENT_PAYED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_PAYED',
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
      self.current_obligation = None
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
      
      return self.get_act(ts_cur)
  
  def set_goal(self) -> None:
    if self.current_obligation != None:
      self.goal = self.get_cur_obl()
    else:
      self.goal = 'apple'
  
  def has_policy(self, ts_cur: AgentTimestep) -> bool:
    s_next = self.hash_ts(ts_cur)
    if s_next in self.V[self.goal].keys():
      return True
    return False
  
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
    new_pos = ts.observation['POSITION']
    self.update_last_actions(ts.observation, actions[idx])
    if not self.is_water(ts.observation, new_pos):
      self.update_surroundings(new_pos, ts.observation, idx)
    ts.observation = self.update_obs_without_coordinates(ts.observation)
    ts.observation['RIOTS'] = self.update_riots(actions, ts.observation)
    self.set_interpolation_and_dirt_fraction(ts.observation)

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
    if not self.exceeds_map(x, y):
      return self.update_observation(obs, x, y)
    return obs

  def update_observation(self, obs, x, y) -> dict:
    """Updates the observation with requested information."""
    obs['NUM_APPLES_AROUND'] = self.get_apples(obs, x, y)
    obs['WATER_LOCATION'] = list(zip(*np.where(obs['SURROUNDINGS'] <= -3)))
    obs['CUR_CELL_HAS_APPLE'] = True if obs['SURROUNDINGS'][x][y] == -1 else False
    lua_idx = obs['PY_INDEX']
    self.make_territory_observation(obs, x, y, lua_idx)
    obs['POSITION_OTHERS'] = self.get_others(obs)
    return obs
  
  def update_last_actions(self, obs: dict, action: int) -> None:
    """Updates the "last done x" section"""
    x, y = obs['POSITION']

    # Create a dictionary to store the counters for each action
    last_counters = {
        7: 'last_zapped',
        8: 'last_cleaned',
        11: 'last_payed'
    }

    for counter_name in last_counters.values():
        setattr(self, counter_name, getattr(self, counter_name) + 1)

    if self.payees == None:
      self.last_payed = 0

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
                    self.last_payed = 0

      elif action == 7:
        for riot in self.riots:
          if self.is_close_to_agent(obs, riot):
            self.last_zapped = 0

    # Update the observations with the updated counters
    self.get_bool_action(observation=obs, action=action)
    obs['SINCE_AGENT_LAST_ZAPPED'] = self.last_zapped
    obs['SINCE_AGENT_LAST_CLEANED'] = self.last_cleaned
    obs['SINCE_AGENT_LAST_PAYED'] = self.last_payed

  def hit_dirt(self, obs, x, y) -> bool:
    for i in range(x-2, x+2):
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
    observation['AGENT_PAYED'] = True if action == 11 else False
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
      cur_pos = list(zip(*np.where(observation['SURROUNDINGS'] == idx+1)))
      if not self.exceeds_map(x, y):
        observation['SURROUNDINGS'][cur_pos[0][0]][cur_pos[0][1]] = 0
        observation['SURROUNDINGS'][x][y] = idx+1

  def increase_action_steps(self, observation: dict) -> None:
      observation['SINCE_AGENT_LAST_ZAPPED'] = observation['SINCE_AGENT_LAST_ZAPPED'] + 1
      observation['SINCE_AGENT_LAST_CLEANED'] = observation['SINCE_AGENT_LAST_CLEANED'] + 1 
      observation['SINCE_AGENT_LAST_PAYED'] = observation['SINCE_AGENT_LAST_PAYED'] + 1

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
          observation['TIME_TO_GET_PAYED'] = 0
        new_pos = cur_pos + self.action_to_pos[orientation][action]
        if self.is_water(observation, new_pos):
          new_pos = cur_pos # don't move to water
        observation['POSITION'] = new_pos
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
          reward, cur_inventory, payed_time = self.compute_eat_pay(action, action_name, 
                                                    cur_inventory, observation)
          observation['SINCE_AGENT_LAST_PAYED'] = payed_time
          if action == 11:
            observation['AGENT_PAYED'] = True
          else:
            observation['AGENT_ATE'] = True

      observation['INVENTORY'] = cur_inventory
      observation['WATER_LOCATION'] = list(zip(*np.where(observation['SURROUNDINGS'] <= -3)))

      next_timestep.step_type = dm_env.StepType.MID
      next_timestep.reward = reward
      next_timestep.observation = observation

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
    last_cleaned_time = observation['SINCE_AGENT_LAST_CLEANED']
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

    payed_time = observation['SINCE_AGENT_LAST_PAYED']
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
                payed_time = 0

    return reward, cur_inventory, payed_time

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
    x_start = observation['POSITION'][0]-1
    y_start = observation['POSITION'][1]-1
    x_stop = observation['POSITION'][0]+2
    y_stop = observation['POSITION'][1]+2

    for i in range(x_start, x_stop):
      for j in range(y_start, y_stop):
        if not self.exceeds_map(i, j):
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
  
  def custom_deepcopy(self, old_obs):
    """Own copy implementation for time efficiency."""
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
  
  def get_cur_obl(self) -> str:
    """
    Returns the a string with the goal of the obligation.
    """
    if "CLEAN" in self.current_obligation.goal:
      return "clean"
    if "PAY" in self.current_obligation.goal:
      return "pay"
    if "RIOTS" in self.current_obligation.goal:
      return "zap"
    
    return None

  def get_ts_hash_key(self, obs: dict, reward: float, goal: str) -> str:
    relevant_keys = self.relevant_keys[goal] # define keys
    items = list(obs[key] for key in sorted(obs.keys()) if key in sorted(relevant_keys)) # extract
    #sorted_items = sorted(items, key=lambda x: x[0])

    list_bytes = pickle.dumps(items + [reward]) # make byte arrays
    hash_key = hashlib.sha256(list_bytes).hexdigest()  # hash

    """if hash_key in self.hash_table:
      existing_obs = self.hash_table[hash_key]
      new_obs = obs
        
      mismatch_found = False  # Flag to indicate if a mismatch is found
      # Compare the observations
      for key in existing_obs.keys():
        if key not in relevant_keys:
          continue
        
        if isinstance(existing_obs[key], np.ndarray):
          if not np.array_equal(existing_obs[key], new_obs[key]):
            print(f"Key: {key}")
            print("Stored Value:", existing_obs[key])
            print("New Value:", new_obs[key])
            mismatch_found = True
        else:
          if existing_obs[key] != new_obs[key]:
            print(f"Key: {key}")
            print("Stored Value:", existing_obs[key])
            print("New Value:", new_obs[key])
            mismatch_found = True
        
      if mismatch_found:
        print(f"Hash collision detected for key {hash_key}")

    else:
      self.hash_table[hash_key] = obs"""

    return hash_key

  def hash_ts(self, ts: AgentTimestep):
    """Computes hash for the given timestep observation."""
    return self.get_ts_hash_key(ts.observation, ts.reward, ts.goal)
  
  # from https://github.com/JuliaPlanners/SymbolicPlanners.jl/blob/master/src/planners/rtdp.jl
  def rtdp(self, ts_start: AgentTimestep) -> None:
    # Perform greedy value iteration
    visited = []
    for _ in range(self.n_rollouts):
      ts_cur = ts_start
      for _ in range(self.max_depth):
        visited.append(ts_cur)
        # greedy rollout giving the next best action
        next_act = self.update(ts_cur)
        # visited += neighbors
        # taking nest best action
        ts_cur = self.env_step(ts_cur, next_act, self.py_index)

    # post-rollout update
    while len(visited) > 0:
      ts_cur = visited.pop()
      _ = self.update(ts_cur)

    return
  
  def get_act(self, ts_cur: AgentTimestep, idx: int, no_rules=False, others=False) -> int:
    hash = self.hash_ts(ts_cur)
    goal = ts_cur.goal

    if no_rules:
      if others:
        v_func = self.all_bots[idx].V_ruleless[goal]
      else:
        v_func = self.V_ruleless[goal]
    else:
      if not no_rules:
        v_func = self.all_bots[idx].V[goal]
      else:
        v_func = self.V[goal]

    if hash in v_func.keys():
      if self.log_output:
        print(f'position: {ts_cur.observation["POSITION"]}, key: {hash}')
        print(v_func[hash])
      next_act = self.get_boltzmann_act(v_func[hash])
      return next_act
    
  """def get_action_prob(self, act: int, ts_cur: AgentTimestep,  no_rules=False) -> float:
    hash = self.hash_ts(ts_cur)
    v_func = self.V[self.goal] if not no_rules else self.V_ruleless[self.goal]
    return v_func[hash][act]"""

  def update(self, ts_cur: AgentTimestep) -> int:
    """Updates state-action pair value function 
    and returns the best action based on that."""
    size = self.action_spec.num_values 
    Q = np.full(size, -1.0)
    Q_ruleless = np.full(size, -1.0)

    # TODO: change to available_action_history()
    available = self.available_actions(ts_cur.observation)    
    s_cur = self.hash_ts(ts_cur)

    if self.log_weights:
      print(f'NEW UPDATE FOR STATE {s_cur}')

    # initialize best optimistic guess for cur state
    if s_cur not in self.V[self.goal].keys():
      self.V[self.goal][s_cur] = self.init_heuristic(ts_cur)
    
    for act in range(self.action_spec.num_values):
      ts_next = self.env_step(ts_cur, act, self.py_index)
      s_next = self.init_process_next_ts(ts_next)

      Q[act], Q_ruleless[act]  = self.get_estimated_return(ts_next, s_next, act, available, type, ts_cur)

    self.V[self.goal][s_cur] = Q
    self.V_ruleless[self.goal][s_cur] = Q_ruleless

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
  
  def init_process_next_ts(self, ts_cur):
    pos = ts_cur.observation['POSITION']
    
    if self.exceeds_map(pos[0], pos[1]):
      return ''

    s_next = self.hash_ts(ts_cur)
    # initialize best optimistic guess for next state
    if s_next not in self.V[self.goal].keys():
      self.V[self.goal][s_next] = self.init_heuristic(ts_cur)

    return s_next
  
  def is_agent_in_position(self, observation: dict, pos) -> bool:
    surroundings = observation['SURROUNDINGS']
    return surroundings[pos[0], pos[1]] > 0 and surroundings[pos[0], pos[1]] != self.lua_index
  
  def init_heuristic(self, timestep: AgentTimestep) -> np.array:
    size = self.action_spec.num_values 
    Q = np.full(size, -np.inf)

    if self.log_weights:
      print()
      print(f"{timestep.observation['POSITION']} for {self.hash_ts(timestep)}")

    for act in range(size):
      ts_next = self.env_step(timestep, act, self.py_index)
      observation = ts_next.observation
      pos = observation['POSITION']
      reward = 0.0

      if self.exceeds_map(pos[0], pos[1]):
        continue

      if self.is_agent_in_position(observation, pos) or self.is_water(observation, pos):
        reward -= self.default_action_cost
    
      if self.goal == "apple":
        pos_eaten_apple = tuple((pos[0], pos[1])) if act == 10 and observation['INVENTORY'] > 0 else 0
        pos_cur_apples = self.get_cur_obj_pos(observation['SURROUNDINGS'], object_idx = -1)
        pos_fut_apples = [apple for apple in self.pos_all_possible_apples if apple not in pos_cur_apples and apple != pos_eaten_apple]

        r_eat_apple = self.apple_reward if act == 10 and observation['INVENTORY'] > 0 else 0
        r_inv_apple = self.apple_reward * observation['INVENTORY'] * self.gamma**(observation['INVENTORY'])
        r_inv_apple -= self.default_action_cost * observation['INVENTORY'] * self.gamma**(observation['INVENTORY'])
        r_cur_apples = self.get_discounted_reward(pos_cur_apples, pos, observation)
        r_fut_apples = self.get_discounted_reward(pos_fut_apples, pos, observation, respawn_type='apple')

        reward = r_cur_apples + r_eat_apple + r_fut_apples + r_inv_apple

        if self.log_weights:
          print(f"len cur_apples: {len(pos_cur_apples)}, reward: {reward}, r_cur_apples: {r_cur_apples}, r_eat_apple: {r_eat_apple}, r_fut_apples: {r_fut_apples}, r_inv_apple: {r_inv_apple}")

      else:
        pos_cur_obl = self.get_cur_obl_pos(observation)

        r_cur_obl = self.get_discounted_reward(pos_cur_obl, pos, observation)
        r_fulfilled_obl = self.obligation_reward if self.current_obligation.satisfied(observation) else 0

        r_fut_obl = 0
        if self.goal == 'clean':
          pos_fut_obl = [dirt for dirt in self.pos_all_possible_dirt if dirt not in pos_cur_obl]
          r_fut_obl = self.get_discounted_reward(pos_fut_obl, pos, observation, respawn_type='dirt')

        reward = r_cur_obl + r_fulfilled_obl + r_fut_obl
        
        if self.log_weights:
          # print(f"len pos_fut_obl: {len(pos_fut_obl)}, reward: {r_fut_obl}, fulfilled: {r_fulfilled_obl}")
          print(f"len pos_cur_obl: {len(pos_cur_obl)}, reward: {r_cur_obl}, fulfilled: {r_fulfilled_obl}")

      Q[act] = reward

    if self.log_weights:
      print(Q)

    return Q
  
  def get_discounted_reward(self, target_pos, own_pos, obs, respawn_type=None) -> float:
    reward = 0.0
    r_amount = self.apple_reward if self.goal == 'apple' else self.obligation_reward
    dirt_conditioned_regrowth_rate = self.get_dirt_conditioned_regrowth_rate()

    for pos in target_pos:
      n_steps_to_reward = int(self.manhattan_dis(pos, own_pos))

      if respawn_type == None: # Consider only currently existing objects
        reward += r_amount * self.gamma**(n_steps_to_reward) # Positive reward for eating apple
        reward -= self.default_action_cost * self.gamma**(n_steps_to_reward) # Cost of eating action
      
      else: # Future objects
        respawn_rate = self.dirt_spawn_prob
        if respawn_type == 'apple':
          regrowth_prob_idx = min(self.get_apples(obs, pos[0], pos[1]), self.num_regrowth_probs-1)
          respawn_rate = dirt_conditioned_regrowth_rate * self.regrowth_probabilities[regrowth_prob_idx]

        reward += r_amount * respawn_rate * self.gamma**(n_steps_to_reward) # Positive reward for eating apple
      
        for i in range(n_steps_to_reward): # Negative reward 
          reward -= self.default_action_cost * respawn_rate * self.gamma**i

    return reward
  
  def get_cur_obl_pos(self, observation: dict) -> list:
    if self.goal == 'clean':
      # return self.pos_all_possible_dirt
      return list(zip(*np.where(observation['SURROUNDINGS'] == -3)))
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
    return abs(pos_cur[0] - pos_goal[0]) + abs(pos_cur[1] - pos_goal[1])
  
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
  
  def compute_boltzmann(self, q_values: list):

    if self.tau == 0:
      probs = np.zeros_like(q_values)
      probs[np.argmax(q_values)] = 1.0
      return probs
    
    max_q_value = np.max(q_values)
    shifted_q_values = q_values - max_q_value
    if not self.tau == 0:
      probs = np.exp(shifted_q_values / self.tau)
    else:
      probs = np.exp(shifted_q_values)
    probs /= probs.sum() # normalized
    
    # Check and handle NaN values
    if np.any(np.isnan(probs)):
        print("Warning: NaN values detected in probabilities. Using uniform distribution.")
        probs = np.ones_like(q_values) / len(q_values)
    
    return probs
  
  def get_boltzmann_act(self, q_values: list) -> int:

    if self.tau == 0:
        return np.argmax(q_values)
    probs = self.compute_boltzmann(q_values)
    action = np.random.choice(len(q_values), p=probs) 
    return action

  def get_estimated_return(self, ts_next: AgentTimestep, s_next: str, act: int, available: list, type: str, ts_cur: AgentTimestep) -> float:
    observation = ts_next.observation
    pos = observation['POSITION']

    r_forward = max(self.V[self.goal][s_next]) * self.gamma
    r_cur = ts_next.reward

    if self.current_obligation != None:
      r_cur = 0
      if self.current_obligation.satisfied(observation):
        r_cur = self.obligation_reward

    cost = self.compliance_cost if act in available else self.violation_cost # rule violation

    if self.is_agent_in_position(observation, pos) or self.is_water(observation, pos):
      r_cur -= self.default_action_cost

    if self.log_weights:
      print()
      print(f'{ts_cur.observation["POSITION"]} for {act} to {ts_next.observation["POSITION"]} gives\t{r_forward} + {r_cur} - {cost}; {s_next}')

    v_rules = r_forward + r_cur - cost
    v_ruleless = r_forward + r_cur

    return v_rules, v_ruleless
  
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

    apple_mask = (surroundings[x_min:x_max, y_min:y_max] == -1)
    apple_count = np.count_nonzero(apple_mask)

    return apple_count
    
  def make_territory_observation(self, observation, x, y, lua_idx):
    """
    Adds values for territory components to the observation dict.
      AGENT_HAS_STOLEN: if the owner of the current cell has stolen
          from the current agent. True for own and free property, too.
      CUR_CELL_IS_FOREIGN_PROPERTY: True if current cell does not
          belong to current agent.
    """
    property_idx = int(observation['PROPERTY'][x][y])
    observation['AGENT_HAS_STOLEN'] = True

    if lua_idx == 0 or property_idx == lua_idx:
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = False

    else:
      observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = True
      if observation['STOLEN_RECORDS'][lua_idx-1] != 1:
        observation['AGENT_HAS_STOLEN'] = False

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
    elif self.current_obligation != None:
      return self.current_obligation.satisfied(
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
