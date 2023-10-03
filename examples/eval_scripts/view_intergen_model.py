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
"""Runs the bots trained in self_play_train.py and renders in pygame.
You must provide experiment_state, expected to be
~/ray_results/PPO/experiment_state_YOUR_RUN_ID.json
"""

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
import itertools

import os

from meltingpot.python import substrate
import csv
import time

import matplotlib.pyplot as plt
import random

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
# from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from meltingpot.python.utils.policies.lambda_rules import CLEANING_RULES, PICK_APPLE_RULES, TERRITORY_RULES
from meltingpot.python.configs.substrates.rule_obeying_harvest__complete import ROLE_SPRITE_DICT, ROLE_TO_INT, INT_TO_ROLE
from meltingpot.python.utils.policies.rule_generation import RuleGenerator

# from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.rule_adjusting_policy import RuleAdjustingPolicy, DEFAULT_INIT_PRIOR, DEFAULT_MAX_LIFE_SPAN
# from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS
generator = RuleGenerator()
POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS = generator.generate_rules_of_length(2)
POTENTIAL_RULES = POTENTIAL_PROHIBITIONS + POTENTIAL_OBLIGATIONS

def main(roles, 
          episodes, 
          num_iteration, 
          rules, 
          env_seed, 
          create_video=False, 
          log_output=False, 
          log_weights=False,
          save_csv=False,
          render=True,
          ):

  level_name = get_name_from_rules(rules)
  substrate_name = f'rule_obeying_harvest_{level_name}'
  num_bots = len(roles)
  age_range = int(DEFAULT_MAX_LIFE_SPAN / (len(roles)-1))
  ages = list(range(0, len(roles)*age_range, age_range))
  # random.shuffle(ages) # that not nearby agents a likely to die
  print(ages)

  config = {'substrate': substrate_name,
            'roles': roles}
  
  env = substrate.build(config['substrate'], 
                        roles=config['roles'], 
                        env_seed=env_seed)

  bots = [] # list of currently active bots
  bot_dicts = [] # stores data of each bot: reward, action, beleifs for every step
  bot_appearance = {} # keeps track of ALL bots, if their active or dead, and their role
  role_str = '' # name for video creation

  """
  Create initial set of bots.
  """

  for i, role in enumerate(roles):
    bots.append(RuleAdjustingPolicy(env=env, 
                                    player_idx=i,
                                    log_output=log_output,
                                    log_rule_prob_output=False,
                                    log_weights=log_weights,
                                    look=ROLE_TO_INT[role],
                                    role=role, 
                                    num_players=num_bots,
                                    potential_prohibitions=POTENTIAL_PROHIBITIONS,
                                    potential_obligations=POTENTIAL_OBLIGATIONS,
                                    active_prohibitions=DEFAULT_PROHIBITIONS,
                                    active_obligations=DEFAULT_OBLIGATIONS,
                                    is_learner=True,
                                    age=ages[i],
                                    ))
    bot_dicts.append(make_empty_dict(POTENTIAL_RULES))
    bot_appearance[i] = [role, i]
      
    for bot in bots:
      bot.set_all_bots(bots)
      
  for role in set(roles):
    role_str += role # video name
    role_str += str(roles.count(role))
    role_str += '_'

  timestep = env.reset()
  dead_apple_ratio = 0.0
  goals = ['apple', 'clean', 'pay', 'zap']

  # actions = {key: [0] for key in range(len(bots))}
  actions = [0] * len(bots)

  # Configure the pygame display
  scale = 4
  fps = 5

  if render:
    pygame.init()
    # start_time = time.time()  # Start the timer
    clock = pygame.time.Clock()
    pygame.display.set_caption("DM Lab2d")
    obs_spec = env.observation_spec()
    shape = obs_spec[0]["WORLD.RGB"].shape
    game_display = pygame.display.set_mode(
        (int(shape[1] * scale), int(shape[0] * scale)))

  """ 
  The below is only for reading in policies from earlier runs. 
  These scale up very fast! 
  """
  for bot in bots:
    filename = f'examples/results/policies/{bot.role}_policies.csv'
    filename_wo_rules = f'examples/results/policies/{bot.role}_policies_wo_rules.csv'
    if os.path.exists(filename):
      bot.V = read_from_csv(filename)
      for goal in goals:
        if not goal in bot.V.keys():
          bot.V[goal] = {} 
    if os.path.exists(filename_wo_rules):
      bot.V_wo_rules = read_from_csv(filename_wo_rules)
      for goal in goals:
        if not goal in bot.V_wo_rules.keys():
          bot.V_wo_rules[goal] = {} 

  """
  Actual game loop.
  """
  for k in range(episodes):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    if render:
      surface = pygame.surfarray.make_surface(obs)
      rect = surface.get_rect()
      surf = pygame.transform.scale(surface,
                                    (int(rect[2] * scale), int(rect[3] * scale)))

      game_display.blit(surf, dest=(0, 0))
      pygame.display.update()
      clock.tick(fps)

    timestep_list = [bot.add_non_physical_info(timestep, actions, i) for i, bot in enumerate(bots)]
    last_actions = np.copy(actions)

    for i, bot in enumerate(bots):            
      bot.append_to_history(timestep_list)
      actions[i] = bot.step()

      if bot.freeze_counter == 1:
        bot.rule_beliefs = [DEFAULT_INIT_PRIOR] * len(bot.rule_beliefs)
        actions = [0] * len(bots) # reset current actions

    for i, bot in enumerate(bots):
      if len(bot.history) > 1:
        bot.update_beliefs(last_actions)
        if k % 10 == 0:
          bot.obligations, bot.prohibitions = bot.sample_rules()
        # bot.obligations, bot.prohibitions = bot.threshold_rules()
    
    dead_apple_ratio = bots[-1].history[-1][-1].observation['DEAD_APPLE_RATIO'] # same for every player
    
    if log_output:
      print('Actions: ' + str(actions)) 

    timestep = env.step(actions)

    """
    Update the data to collect.
    """
    for i, bot in enumerate(bots):
      data_dict = bot_dicts[i]
      is_frozen = bot.freeze_counter > 0
      new_data_dict = append_to_dict(data_dict=data_dict, 
                                reward=timestep.reward[i].item(), 
                                is_frozen=is_frozen,
                                beliefs=bot.rule_beliefs, 
                                action=actions[i],
                                dead_apple_ratio=dead_apple_ratio)
      bot_dicts[i] = new_data_dict

    # Saving files in superdircetory
    if create_video:
      filename = '../videos/screen_%04d.png' % (k)
      pygame.image.save(game_display, filename)

  name = f'vers{num_iteration}_{role_str}'[:-1]
  filename = 'videos/evals/' + name + '.mov'

  if create_video:
    make_video(filename)

  """
  Saving computed policies
  """
  if save_csv:
    for i, bot in enumerate(bots):
      filename = f'examples/results/policies/{bot.role}_policies.csv'
      filename_wo_rules = f'examples/results/policies/{bot.role}_policies_wo_rules.csv'
      save_to_csv(filename, bot.V)
      save_to_csv(filename_wo_rules, bot.V_wo_rules)

  settings = get_settings(bots=bots, rules=rules)

  key_with_max_value = max(bots[0].hash_count, key=bots[0].hash_count.get)
  print(f"The key with the highest value is: {key_with_max_value}")
  print(f"Its value is: {bots[0].hash_count[key_with_max_value]}")
  # end_time = time.time()  # End the timer

  # return end_time - start_time, data_dict
  return settings, bot_dicts

  """ Profiler Run:
  ~ python3 -m cProfile -o output.prof examples/evals/evals.py 
  snakeviz output.prof 
  """

def make_empty_dict(potential_rules) -> dict:
  data_dict = {
    'reward': [],
    'action': [],
    'dead_apple_ratio': [],
    'is_frozen': [],
    }
  
  for rule in potential_rules:
    data_dict[rule.make_str_repr()] = []

  return data_dict

def get_most_frequent_states(q_value_log, top_k=10):
    states = []
    sorted_states = sorted(q_value_log.items(), key=lambda x: x[1], reverse=True)
    for i, (state, count) in enumerate(sorted_states[:top_k]):
      states.append(state)
    return states

# Function to save the bot.V nested dictionaries to a CSV file
def save_to_csv(filename, data):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['Goal', 'Hashkey'] + [f'Value {i + 1}' for i in range(12)])

    # Write the data rows
    for goal, hash_key_or_dict in data.items():
      if type(hash_key_or_dict) == dict:
        for hashkey, values in hash_key_or_dict.items():
          row_data = [str(goal), str(hashkey)] + [str(val) for val in values]
          writer.writerow(row_data)
      else:
        row_data = ['None', str(goal)] + [str(hash_key_or_dict[i]) for i in range(len(hash_key_or_dict))]
        writer.writerow(row_data)

# Function to read the CSV file and create the nested dictionary
def read_from_csv(filename):
  data = {}
  with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row

    for row in reader:
      goal, hashkey = row[:2]
      values = [float(val) for val in row[2:]]
      
      if goal == 'None':  # This indicates that hashkey_or_dict was not a dict
          data[hashkey] = values
          continue

      if goal not in data:
          data[goal] = {}
      data[goal][hashkey] = values

  return data

def append_to_dict(data_dict: dict, reward, beliefs, is_frozen, action, dead_apple_ratio) -> dict:
  """
    Appends reward and action data to the given dictionary for different roles.

    Parameters:
    - data_dict (dict):   The dictionary to which the data will be appended.
    - reward_arr:         The array containing reward values.
    - beliefs:            Array of posteriors about potential rules at that timestep.
    - actions:            The array containing action values.
    - dead_apple_ratio:   Ratio of apples that won't grow again.

    Returns:
    - dict: The updated data dictionary.

    Note:
    - The first four keys in `data_dict` must relate to player rewards.
    - The next four keys in `data_dict` must relate to player actions.
    - If there are two 'free' agents, the second one's data will be appended to the 'learner' key.
    """
  data_dict['reward'].append(reward)
  data_dict['action'].append(action)
  data_dict['dead_apple_ratio'].append(dead_apple_ratio)
  data_dict['is_frozen'].append(is_frozen)

  for i, key in enumerate(data_dict):
    if i > 3:
      data_dict[key].append(beliefs[i-4]) # get beliefs (start at indec 0)

  return data_dict

def get_index(role, all_roles, skip_first: bool):
  for i, name in enumerate(all_roles):
    if name == role:
      if skip_first == True:
        skip_first = False
        continue
      return i
    
def get_name_from_rules(rules: list) -> str:
  components = set()
  for rule in rules:
    rule_str = rule.make_str_repr()
    if rule_str in CLEANING_RULES:
      components.add("cleaning")
    elif rule_str in PICK_APPLE_RULES:
      components.add("apples")
    elif rule_str in TERRITORY_RULES:
      components.add("territory")

  if len(components) == 0:
    return "_empty"
  elif len(components) == 3:
    return "_complete"
  else:
    output = ''
    for item in sorted(list(components)):
      output += f"_{item}"

  return output

def split_rules(rules):
  obeyed_prohibitions = []
  obeyed_obligations = []
  for rule in rules:
    if isinstance(rule, ProhibitionRule):
      obeyed_prohibitions += [rule]

    elif isinstance(rule, ObligationRule):
      obeyed_obligations += [rule]

  return obeyed_prohibitions, obeyed_obligations

def get_settings(bots, rules):
  settings = []
  for role in ROLE_SPRITE_DICT:
    for agent in bots:
      settings.append(1) if agent.role == role else settings.append(0)

  for rule in DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS:
    settings.append(1) if rule in rules else settings.append(0)

  return settings

def make_video(filename):
    print('\nCreating video.\n')
    os.system('ffmpeg -r 10 -f image2'
              + ' -s 400x400'
              + ' -i ../videos/screen_%04d.png'
              + ' -vcodec libx264 ' 
              + ' -y '
              + filename)


if __name__ == "__main__":
  roles = ("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 1
  episodes = 200
  # Possible values for tau and gamma you want to test
  """taus = [0.0, 0.1, 0.2, 0.3]
  gammas = [0.99999]
  num_runs = 10
  counter = 1
  
  # Initialize results
  results = {(gamma, tau): {'cleaner': 0, 'farmer': 0, 'free': 0} for gamma in gammas for tau in taus}
  
  for run in range(num_runs):
      print(f"Run number: {run+1}")
      for gamma in gammas:
        for tau in taus:

          print()
          print(f'RUN NUMBER {counter}')

          counter += 1"""
    
  settings, data_dict = main(roles=roles,
                                    rules=DEFAULT_RULES,
                                    env_seed=1,
                                    episodes=episodes,
                                    num_iteration=1,
                                    create_video=False,
                                    log_output=False,
                                    log_weights=False,
                                    save_csv=False,
                                    )

  """results[(gamma, tau)]['cleaner'] += sum(data_dict['cleaner'])
      #results[(gamma, tau)]['farmer'] += sum(data_dict['farmer'])
      #results[(gamma, tau)]['free'] += sum(data_dict['free'])

  # Calculate averages
  for (gamma, tau), scores in results.items():
      scores['cleaner'] /= num_runs
      scores['farmer'] /= num_runs
      scores['free'] /= num_runs
  
  # Print or process averaged results as desired
  for (gamma, tau), scores in results.items():
      print(f"For gamma={gamma}:")
      print(f"cleaner: {scores['cleaner']:.2f}, farmer: {scores['farmer']:.2f}, free: {scores['free']:.2f}")
      print()
"""
  #for item, value in data_dict.items():
  #print(f'{item}: {value}')
  #print(data_dict)
