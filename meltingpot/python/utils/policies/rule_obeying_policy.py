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

  def __init__(self, 
               env: dm_env.Environment, 
               player_idx: int,
               log_output: bool,
               look: shapes,
               role: str = "free",
               prohibitions: list = DEFAULT_PROHIBITIONS, 
               obligations: list = DEFAULT_OBLIGATIONS) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """

    # CALLING PARAMETER
    self._index = player_idx
    self.role = role
    self.look = look
    self.log_output = log_output
    self.action_spec = env.action_spec()[0]
    self.prohibitions = prohibitions
    self.obligations = obligations

    # HYPERPARAMETER
    self.max_depth = 20
    self.compliance_cost = 0.1
    self.violation_cost = 0.4
    self.tau = 0.1
    self.action_cost = 1
    # self.epsilon = 0.2
    self.regrowth_rate = 0.5
    self.n_steps = 10
    self.gamma = 0.98
    self.n_rollouts = 8
    self.obligation_reward = 1
    
    # GLOBAL INITILIZATIONS
    self.history = deque(maxlen=10)
    self.payees = []
    self.riots = []
    self.pos_all_apples = []
    self.hash_table = {}
    if self.role == 'farmer':
      self.payees = None
    # TODO condition on set of active rules
    self.V = {'apple': {}, 'clean': {}, 'pay': {}, 'zap': {}} # nested policy dict
    self.ts_start = None
    self.goal = None
    self.x_max = 15
    self.y_max = 15
    self.old_pos = None

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
    
    self.relevant_apple_keys = [
      'POSITION', 
      'ORIENTATION',
      'NUM_APPLES_AROUND',
      # 'POSITION_OTHERS',
      'INVENTORY',
      # 'SINCE_AGENT_LAST_CLEANED',
      'CUR_CELL_IS_FOREIGN_PROPERTY', 
      'CUR_CELL_HAS_APPLE', 
      # 'AGENT_CLEANED'
      ]
    
    self.relevant_obligation_keys = [
      'POSITION', 
      'ORIENTATION',
      'NUM_APPLES_AROUND',
      'POSITION_OTHERS',
      # 'INVENTORY',
      'SINCE_AGENT_LAST_CLEANED',
      'SINCE_AGENT_LAST_PAYED',
      'CUR_CELL_IS_FOREIGN_PROPERTY', 
      'CUR_CELL_HAS_APPLE', 
      'AGENT_CLEANED'
      ]
        
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
        self.pos_all_apples = list(zip(*np.where(timestep.observation['SURROUNDINGS']== -3)))

      ts_cur = self.add_non_physical_info(timestep=timestep, actions=actions, idx=self._index)
      self.ts_start = ts_cur

      # Check if any of the obligations are active
      self.current_obligation = None
      for obligation in self.obligations:
         if obligation.holds_in_history(self.history):
           self.current_obligation = obligation
           break
         
      self.set_goal()
               
      if self.log_output:
        print(f"player: {self._index} obligation active?: {self.current_obligation != None}")

      if not self.has_policy(ts_cur):
        self.rtdp(ts_cur)
      
      # return [self.get_best_act(ts_cur)]
      return self.get_optimal_path(ts_cur)
  
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
    dict_observation = self.deepcopy_dict(timestep.observation[idx])
    for obs_name, obs_val in dict_observation.items():
      ts.add_obs(obs_name=obs_name, obs_val=obs_val)

    # not sure whether to subtract 1 or not
    ts.observation['POSITION'][0] = ts.observation['POSITION'][0]-1
    ts.observation['POSITION'][1] = ts.observation['POSITION'][1]-1
    new_pos = ts.observation['POSITION']
    self.update_last_actions(ts.observation, actions[idx])
    if not self.is_water(ts.observation, new_pos):
      self.update_surroundings(new_pos, ts.observation, idx)
    ts.observation = self.update_obs_without_coordinates(ts.observation)
    ts.observation['RIOTS'] = self.update_riots(actions, ts.observation)

    return ts
  
  def update_riots(self, actions: list, obs: dict) -> list:
    for i, action in enumerate(actions):
      if action == 7:
        player_who_zapped = i
        zapped_agent = self.get_zapped_agent(player_who_zapped, obs)
        if zapped_agent in self.riots:
          self.riots = self.riots.remove(zapped_agent)
    
    return self.riots

  def get_zapped_agent(self, player_who_zapped: int, obs: dict) -> int:
    print('player_who_zapped: ' + str(player_who_zapped))
    print('position others: ' + str(obs['POSITION_OTHERS']))
    x, y = obs['POSITION_OTHERS'][player_who_zapped][0], obs['POSITION_OTHERS'][player_who_zapped][1]
    for i in range(x-1, x+2):
      for j in range(y-1, y+2):
        if not self.exceeds_map(i, j):
          if obs['SURROUNDINGS'][i][j] > 0:
            if obs['SURROUNDINGS'][i][j] != player_who_zapped:
              return obs['SURROUNDINGS'][i][j]
  
  def update_obs_without_coordinates(self, obs: dict) -> dict:
    cur_pos = np.copy(obs['POSITION'])
    x, y = cur_pos[0], cur_pos[1]
    if not self.exceeds_map(x, y):
      return self.update_observation(obs, x, y)
    return obs

  def update_observation(self, obs, x, y) -> dict:
    """Updates the observation with requested information."""
    obs['NUM_APPLES_AROUND'] = self.get_apples(obs, x, y)
    obs['CUR_CELL_HAS_APPLE'] = True if obs['SURROUNDINGS'][x][y] == -3 else False
    self.make_territory_observation(obs, x, y)
    obs['POSITION_OTHERS'] = self.get_others(obs)
    # break
    return obs
  
  def update_last_actions(self, obs: dict, action: int) -> None:
    """Updates the "last done x" section"""
    # Create a dictionary to store the counters for each action
    last_counters = {
        7: 'last_zapped',
        8: 'last_cleaned',
        11: 'last_payed'
    }

    for counter_name in last_counters.values():
        setattr(self, counter_name, getattr(self, counter_name) + 1)

    if action in last_counters:
      setattr(self, last_counters[action], 0)

    # Update the observations with the updated counters
    obs['EAT_ACTION'] = True if action == 10 else False
    obs['SINCE_AGENT_LAST_ZAPPED'] = self.last_zapped
    obs['SINCE_AGENT_LAST_CLEANED'] = self.last_cleaned
    obs['SINCE_AGENT_LAST_PAYED'] = self.last_payed
  
  def get_others(self, observation: dict) -> list:
    """Returns the indices of all players in a 2D array."""
    surroundings = observation['SURROUNDINGS']
    positive_values_mask = (surroundings > 0)
    positive_values = surroundings[positive_values_mask]
    sorted_indices = np.argsort(positive_values)
    sorted_positive_indices = np.argwhere(positive_values_mask)[sorted_indices]
    return [(index[0], index[1]) for index in sorted_positive_indices]
  
  def update_and_append_history(self, timestep: dm_env.TimeStep, actions: list) -> None:
    """Append current timestep obsetvation to observation history."""
    ts_cur = self.add_non_physical_info(timestep, actions, self._index)
    self.history.append(ts_cur.observation)

  def maybe_collect_apple(self, observation) -> float:
    x, y = observation['POSITION'][0], observation['POSITION'][1]
    reward_map = observation['SURROUNDINGS']
    has_apple = observation['CUR_CELL_HAS_APPLE']
    if self.exceeds_map(x, y):
      return 0, has_apple
    if reward_map[x][y] == -3:
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
      observation = self.deepcopy_dict(timestep.observation)
      observation = self.update_obs_without_coordinates(observation)
      self.increase_action_steps(observation)
      next_timestep = AgentTimestep()
      orientation = observation['ORIENTATION']
      observation['AGENT_CLEANED'] = False
      observation['EAT_ACTION'] = True if action == 10 else False
      cur_inventory = observation['INVENTORY']
      cur_pos = observation['POSITION']
      reward = 0
      action_name = None

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
          observation['RIOTS'] = riots

        if action_name == 'CLEAN_ACTION':
          last_cleaned_time, num_cleaners = self.compute_clean(observation, x, y)
          observation['SINCE_AGENT_LAST_CLEANED'] = last_cleaned_time
          observation['AGENT_CLEANED'] = True
          observation['TOTAL_NUM_CLEANERS'] = num_cleaners

        if cur_inventory > 0: # EAT AND PAY
          reward, cur_inventory, payed_time = self.compute_eat_pay(action, action_name, 
                                                    cur_inventory, observation)
          observation['SINCE_AGENT_LAST_PAYED'] = payed_time

      observation['INVENTORY'] = cur_inventory
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
          riots = riots.remove(riot)

    return last_zapped, riots

  def compute_clean(self, observation, x, y):
    last_cleaned_time = observation['SINCE_AGENT_LAST_CLEANED']
    num_cleaners = observation['TOTAL_NUM_CLEANERS']
    if not self.role == 'farmer':
      # if facing north and is at water
      if not self.exceeds_map(x, y):
        if observation['ORIENTATION'] == 0:
          if observation['SURROUNDINGS'][x][y-1] == -1 or observation['SURROUNDINGS'][x][y-2] == -1:
            last_cleaned_time = 0
            num_cleaners = 1

    return last_cleaned_time, num_cleaners
  
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
          if self.payees == None:
            self.payees = self.get_payees(observation)
          for payee in self.payees:
            if self.is_close_to_agent(observation, payee):
              cur_inventory -= 1 # pay
              payed_time = 0

    return reward, cur_inventory, payed_time

  def get_payees(self, observation):
    payees = []
    if isinstance(observation['ALWAYS_PAYING_TO'], np.int32):
      payees.append(observation['ALWAYS_PAYING_TO'])
    else:
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
  
  def deepcopy_dict(self, old_obs):
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

  def get_ts_hash_key(self, obs, reward):
    # Convert the dictionary to a tuple of key-value pairs
    relevant_keys = self.relevant_apple_keys if self.current_obligation == None else self.relevant_obligation_keys
    items = tuple((key, value) for key, value in obs.items() if key in relevant_keys)
    sorted_items = sorted(items, key=lambda x: x[0])
    # items = [value[1] for value in sorted_items]
    list_bytes = pickle.dumps(sorted_items + [reward])
    hash_key = hashlib.sha256(list_bytes).hexdigest() 
    return hash_key
  
  def hash_ts(self, timestep: dm_env.TimeStep):
    """Encodes the state, action pairs and saves them in a hash table."""
    hash_key = self.get_ts_hash_key(timestep.observation, timestep.reward)
    self.hash_table[hash_key] = (timestep)
    return hash_key
  
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
        print(f'NEXT ACTION: {next_act}')
        # taking nest best action
        ts_cur = self.env_step(ts_cur, next_act, self._index)

    # post-rollout update
    while len(visited) > 0:
      ts_cur = visited.pop()
      _ = self.update(ts_cur)

    return
  
  def get_best_act(self, ts_cur: AgentTimestep) -> int:
    hash = self.hash_ts(ts_cur)
    if hash in self.V[self.goal].keys():
      print(f'position: {ts_cur.observation["POSITION"]}')
      print(self.V[self.goal][hash])
      return np.argmax(self.V[self.goal][hash]) # no null action
    else:
      best_act, _ = self.update(ts_cur)
      return best_act

  def update(self, ts_cur: AgentTimestep) -> int:
    """Updates state-action pair value function 
    and returns the best action based on that."""
    size = self.action_spec.num_values 
    Q = np.full(size, -1.0)

    # TODO: change to available_action_history()
    available = self.available_actions(ts_cur.observation)
    s_cur = self.hash_ts(ts_cur)

    # initialize best optimistic guess for cur state
    if s_cur not in self.V[self.goal].keys():
        self.V[self.goal][s_cur] = self.init_heuristic(ts_cur)
    
    for act in range(size): 
      ts_next = self.env_step(ts_cur, act, self._index)

      pos = ts_next.observation['POSITION']
      if self.exceeds_map(pos[0], pos[1]):
        continue

      s_next = self.hash_ts(ts_next)
      # initialize best optimistic guess for next state
      if s_next not in self.V[self.goal].keys():
        self.V[self.goal][s_next] = self.init_heuristic(ts_next)

      Q[act]  = self.get_estimated_return(ts_next, s_next, act, available)

    self.V[self.goal][s_cur] = Q
    return self.select_action(Q)
  
  def init_heuristic(self, timestep: AgentTimestep) -> np.array:
    size = self.action_spec.num_values 
    Q = np.full(size, -1.0)

    for act in range(size):
      ts_next = self.env_step(timestep, act, self._index)
      observation = ts_next.observation
      pos = observation['POSITION']
      reward = 0.0
      
      if self.exceeds_map(pos[0], pos[1]):
        continue

      if self.goal == "apple":
        r_cur_apples = self.get_discounted_reward(self.pos_all_apples, pos)
        r_eaten_apples = 10 if tuple(pos) in self.pos_all_apples and observation['INVENTORY'] != 0 and act == 10 else 0
        reward = r_cur_apples + r_eaten_apples

      else:
        pos_cur_obl = self.get_cur_obl_pos(observation)
        reward = self.get_discounted_reward(pos_cur_obl, pos)

      Q[act] = reward

    return Q
  
  def get_cur_obl_pos(self, observation: dict) -> list:
    if self.goal == 'clean':
      return list(zip(*np.where(observation['SURROUNDINGS'] == -3)))
    elif self.goal == 'pay':
      return  list(zip(*np.where(observation['SURROUNDINGS'] == observation['ALWAYS_PAYING_TO'])))
    else: # self.goal == 'zap'
      return list(zip(*np.where(observation['SURROUNDINGS'] == self.riots)))

  def get_discounted_reward(self, target_pos, own_pos) -> float:
    reward = 0.0
    for pos in target_pos:
      reward += 1 - self.manhattan_dis(pos, own_pos)
    return reward

  def manhattan_dis(self, pos_cur, pos_goal) -> int:
    return abs(pos_cur[0] - pos_goal[0]) + abs(pos_cur[1] - pos_goal[1])

  def get_cur_apples(self, surroundings: np.array) -> list:
    return list(zip(*np.where(surroundings== -3)))
  
  def select_action(self, q_values: list) -> int:
    """if random.random() < self.epsilon:
        action = random.choice(range(self.action_spec.num_values))
    else: # boltzman
      action = np.argmax(q_values)"""
    exp_q_values = np.exp(q_values / self.tau)
    probs = exp_q_values / np.sum(exp_q_values)
    action = np.argmax(probs)

    return action

  def get_estimated_return(self, ts_next: AgentTimestep, s_next: str, act: int, available: list) -> float:
    r_forward = max(self.V[self.goal][s_next]) / self.gamma
    # print(f'first: {max(self.V[self.goal][s_next])}, after scaling: {r_forward}')
    r_cur = ts_next.reward * 10

    if self.current_obligation != None:
      r_cur = 0
      if self.current_obligation.satisfied(ts_next.observation):
        r_cur = self.obligation_reward

    cost = self.compliance_cost if act in available else self.violation_cost # rule violation

    return r_forward + r_cur - cost
  
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
    observation = self.deepcopy_dict(obs)
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

    apple_mask = (surroundings[x_min:x_max, y_min:y_max] == -3)
    apple_count = np.count_nonzero(apple_mask)

    return apple_count
    
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

  def is_water(self, observation, pos):
    x, y = pos[0], pos[1]
    if self.exceeds_map(x, y):
      return True
    if observation["SURROUNDINGS"][x][y] == -1:
      return True
    if observation["SURROUNDINGS"][x][y] == -2:
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
