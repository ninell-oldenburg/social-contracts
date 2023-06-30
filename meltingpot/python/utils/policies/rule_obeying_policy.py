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

from queue import PriorityQueue

from collections import deque

import numpy as np
import itertools
import random
import hashlib

from meltingpot.python.utils.policies import policy
from meltingpot.python.utils.substrates import shapes
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

class EnumPriorityQueue:
  def __init__(self):
    self.queue = PriorityQueue()
    self.elements = []

  def put(self, priority, item):
    self.queue.put((priority, item))
    self.elements.append(item)

  def get(self):
    _, item = self.queue.get()
    self.elements.remove(item)
    return item

  def empty(self):
    return self.queue.empty()
  
  def enumerate(self):
    return iter(self.elements)

  def __contains__(self, item):
    return item in self.elements

  def __iter__(self):
    return iter(self.elements)

  def __len__(self):
    return len(self.elements)
    
class AgentTimestep():
  def __init__(self) -> None:
    self.reward = 0
    self.observation = {}

  def get_obs(self):
    return self.observation
  
  def get_r(self):
    return self.reward
  
  def add_obs(self, obs_name: str, obs_val) -> None:
    self.observation[obs_name] = obs_val
    return
    

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
    self._index = player_idx
    self.role = role
    self.look = look
    self.max_depth = 75
    self.log_output = log_output
    self.action_spec = env.action_spec()[0]
    self.prohibitions = prohibitions
    self.obligations = obligations
    self.is_obligation_active = False
    self.compliance_cost = 1
    self.violation_cost = 10
    self.n_rollouts = 10
    self.hash_table = {}
    self.h_vals = {}
    self.rollout_noise = 0.0
    self.action_noise = 0.0 # change to make probabilistic 
    self.default_heuristic_value = 10
    self.history = deque(maxlen=10)
    self.payees = []
    if self.role == 'farmer':
      self.payees = None

    self.policy_table = {'None': {}} # nested policy dict
    for obligation in self.obligations:
      self.policy_table[obligation] = {}

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
    
    self.relevant_keys = [
      'POSITION', 
      'ORIENTATION', 
      'NUM_APPLES_AROUND', # bit vector
      # position of other agents
      'INVENTORY', # bit vector
      'CUR_CELL_IS_FOREIGN_PROPERTY', 
      'CUR_CELL_HAS_APPLE', 
      'AGENT_CLEANED', 
      'READY_TO_SHOOT'
      ]
        
  def step(self, 
           timestep: dm_env.TimeStep
           ) -> list:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """

      ts_cur = self.add_non_physical_info(timestep=timestep)
      s_cur = self.hash_ts_and_action(timestep=ts_cur)

      # Check if any of the obligations are active
      self.current_obligation = None
      for obligation in self.obligations:
         if obligation.holds_in_history(self.history, self.look):
           self.current_obligation = obligation
           break
      
      if self.log_output:
        print(f"player: {self._index} obligation active?: {self.is_obligation_active}")

      if not self.has_policy(s_cur):
        self.rtdp(s_cur)
      
      return self.f_action_val(s_cur)
  
  def add_non_physical_info(self, timestep: dm_env.TimeStep):
    ts = AgentTimestep()

    for obs_name, obs_val in timestep.observation.iteritems():
      ts.add_obs(obs_name=obs_name, obs_val=obs_val)

    ts.obs = self.update_obs_without_coordinates(ts.observation)
    return ts
  
  def update_obs_without_coordinates(self, obs):
    cur_pos = np.copy(obs['POSITION'])
    x, y = cur_pos[0], cur_pos[1]
    return self.update_observation(obs, x, y)

  def update_observation(self, obs, x, y):
    """Updates the observation with requested information."""
    obs['NUM_APPLES_AROUND'] = self.get_apples(obs, x, y)
    obs['CUR_CELL_HAS_APPLE'] = True if obs['SURROUNDINGS'][x][y] == -3 else False
    self.make_territory_observation(obs, x, y)

    return obs

  def f_action_val(self, state: str) -> list:
    # rule_set = self.policy_table[self.active_rules]
    if self.current_obligation != None:
      condition_dict = self.policy_table[self.current_obligation]
    else: # no obligations active
      condition_dict = self.policy_table['None']
    return np.argmax(condition_dict(state))
  
  def update_and_append_history(self, timestep: dm_env.TimeStep) -> None:
    """Append current timestep obsetvation to observation history."""
    own_cur_obs = self.deepcopy_dict(timestep.observation)
    own_cur_pos = np.copy(own_cur_obs['POSITION'])
    own_x, own_y = own_cur_pos[0], own_cur_pos[1]
    updated_obs = self.update_observation(own_cur_obs, own_x, own_y)
    self.history.append(updated_obs)

  def maybe_collect_apple(self, observation) -> float:
    x, y = observation['POSITION'][0], observation['POSITION'][1]
    reward_map = observation['SURROUNDINGS']
    has_apple = observation['CUR_CELL_HAS_APPLE']
    if self.exceeds_map(observation['WORLD.RGB'], x, y):
      return 0, has_apple
    if reward_map[x][y] == -3:
      return 1, False
    return 0, has_apple

  def env_step(self, timestep: dm_env.TimeStep, action: int) -> dm_env.TimeStep:
      # 1. Unpack observations from timestep
      observation = self.deepcopy_dict(timestep.observation)
      observation = self.update_obs_without_coordinates(observation)
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
        new_inventory, has_apple = self.maybe_collect_apple(observation)
        observation['CUR_CELL_HAS_APPLE'] = has_apple
        cur_inventory += new_inventory

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
      
  def reconstruct_path(self, state: int) -> list:
    """Reconstructs path from path dictionary."""
    _, action = self.unhash(state)
    path = np.array([action])
    while state in self.came_from.keys():
      if state == self.came_from[state]:
        break
      state = self.came_from[state]
      _, action = self.unhash(state)
      path = np.append(path, action)

    path = np.flip(path)
    return path[1:] # first element is always 0
  
  def manhattan_dis(self, s_cur, s_goal):
    """Calculates Manhattan Distance"""
    cur_pos = s_cur
    goal_pos = s_goal
    distance = abs(cur_pos[0] - goal_pos[0]) + abs(cur_pos[1] - goal_pos[1])
    return distance

  def heuristic(self, s_cur):
    """Calculates the heuristic for path search. """
    if self.s_goal:
        for s in self.s_goal:
          ts_cur, _ = self.unhash(s_cur)
          ts_goal, _ = self.unhash(s)
          cur_manhattan = self.manhattan_dis(ts_cur.observation['POSITION'], ts_goal.observation['POSITION'])
          if not s_cur in self.h_vals.keys() or cur_manhattan < self.h_vals[s_cur]:
            return cur_manhattan
    
    # If the hypothetical goal state is not defined, return a default heuristic value
    return self.default_heuristic_value
  
  def cost(self, s_cur, s_bar, action):
    cost = 0
    ts_cur, _ = self.unhash(s_cur)
    ts_goal, _ = self.unhash(s_bar)
    distance = self.manhattan_dis(ts_cur.observation['POSITION'], ts_goal.observation['POSITION'])
    cost += distance

    # check for prohibitions
    available = self.available_actions(ts_cur.observation)
    if action in available:
      cost += self.compliance_cost
    else:
      cost += self.violation_vost

    return cost

  def unhash(self, hash_val):
    timestep = self.hash_table[hash_val][0]
    action = self.hash_table[hash_val][1]
    return timestep, action
  
  def get_hash_key(self, obs, action):
    # Convert the dictionary to a tuple of key-value pairs
    items = tuple((key, value, action) for key, value in obs.items() if key in self.relevant_keys)
    sorted_items = sorted(items, key=lambda x: x[0])
    hash_key = hashlib.sha256(str(sorted_items).encode()).hexdigest() 
    return hash_key
  
  def hash_ts_and_action(self, timestep: dm_env.TimeStep, action: int):
    hash_key = self.get_hash_key(timestep.observation, action)
    self.hash_table[hash_key] = (timestep, action)
    return hash_key
  
  def cal_h_value(self, PRIO_QUEUE, S_VISITED, g_table, came_from):
        queue_values = {}
        # iter through every state left in the queue
        for s in PRIO_QUEUE.enumerate():
            # update h values of the queue with preceding g value and current h_value
            dis = self.manhattan_dis(s, came_from[s])
            queue_values[s] = g_table[came_from[s]] + dis + self.h_vals[s]
        min_s_queue = min(queue_values, key=queue_values.get)
        f_min = queue_values[min_s_queue]
        # update h_vals based on minimum
        for s in S_VISITED: # only visits 
            self.h_vals[s] = f_min - g_table[s]

        return min_s_queue
  
  def get_neighbors(self, s):
    neighbors = []
    cur_ts, _ = self.unhash(s)
    for action in range(self.action_spec.num_values):
      neighbors_ts = self.env_step(cur_ts, action)
      s_next = self.hash_ts_and_action(neighbors_ts, action)
      neighbors.append(s_next)

    return neighbors
  
  def update_s_cur(self, s_cur, s_next, came_from):
    _, action = self.unhash(s_next)
    path = np.array([action])

    while s_next in came_from.keys():

      # TODO: make the actions h_value relevant
      """h_list = {}
      s_neighbors = self.get_neighbors(s_next)
      for s_n in s_neighbors:
        if s_n in self.h_vals.keys():
          h_list[s_n] = self.h_vals[s_n]"""

      s_next = came_from[s_next]
      _, action = self.unhash(s_cur)
      path = np.append(path, action)
      
      if s_next == s_cur:
        path = np.flip(path)
        return s_next, path[1:] # first element is always 0

  # source: http://idm-lab.org/bib/abstracts/papers/aamas06.pdf
  # used a lot from: https://github.com/zhm-real/PathPlanning/blob/master/Search_based_Planning/Search_2D/RTAAStar.py#L42
  def real_time_adaptive_astar(self, timestep: dm_env.TimeStep) -> list[int]:
    timestep = timestep._replace(reward=0.0)
    init_action = 0
    s_cur = self.hash_ts_and_action(timestep, init_action)
    n_rollouts = 0

    while s_cur not in self.s_goal:
      PRIO_QUEUE, S_VISITED, g_vals, came_from = self.a_star(s_cur)

      if PRIO_QUEUE == True: # terminal state of A*
        return self.reconstruct_path(S_VISITED)

      # s_next is the next cheapest node
      s_next = self.cal_h_value(PRIO_QUEUE, S_VISITED, g_vals, came_from)
      s_cur, path_k = self.update_s_cur(s_cur, s_next, came_from)

      if n_rollouts >= self.n_rollouts:
        return path_k
      
      count_searches += 1
    
    return
  
  def rtdp(self, state: str):
    pass
  
  def a_star(self, s_start: int) -> list[int]:
    """Perform a A* search to generate plan."""
    PRIO_QUEUE, S_ORDERED, action = EnumPriorityQueue(), [], 0
    PRIO_QUEUE.put(0, s_start) # ordered by reward
    self.h_vals[s_start] = self.heuristic(s_start)
    came_from = {s_start: s_start}
    g_table = {s_start: 0}
    for s_goal in self.s_goal: # g_vals for current goals
      g_table[s_goal] = float("inf")
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
        s_next = self.hash_ts_and_action(next_timestep, action)

        if s_next not in S_ORDERED: # has not been visited yet
          new_cost = g_table[s_cur] + self.cost(s_cur, s_next, action)
          if s_next not in g_table:
            g_table[s_next] = float("inf")
          if new_cost < g_table[s_next]:  # conditions for updating cost
            g_table[s_next] = new_cost
            came_from[s_next] = s_cur

          self.h_vals[s_next] = self.heuristic(s_next)
          priority = g_table[s_next] + self.h_vals[s_next]
          PRIO_QUEUE.put(priority=priority, item=s_next)

      if depth >= self.max_depth:
        break

    return PRIO_QUEUE, S_ORDERED, g_table, came_from
  
  def intersection(self, lst1, lst2):
    out = [value for value in lst1 if value in lst2]
    return out
  
  def available_actions_history(self):
    action_list = self.available_actions(self.history[0])
    for obs in self.history[1:]:
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

      if self.exceeds_map(observation['WORLD.RGB'], x, y):
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

  def is_done(self, state: str):
    """Check whether any of the break criteria are met."""
    timestep, action = self.unhash(state)
    if timestep.last():
      return True
    elif self.current_obligation != None:
      return self.current_obligation.satisfied(
        timestep.observation, self.look)
    elif timestep.reward >= 1.0:
      return True
    return False

  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
