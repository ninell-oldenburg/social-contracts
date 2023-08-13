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

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from meltingpot.python.utils.policies.lambda_rules import CLEANING_RULES, PICK_APPLE_RULES, TERRITORY_RULES
from meltingpot.python.configs.substrates.rule_obeying_harvest__complete import ROLE_SPRITE_DICT
from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.rule_adjusting_policy import RuleAdjustingPolicy
from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS

def main(roles, 
          episodes, 
          num_iteration, 
          rules, 
          env_seed, 
          create_video=True, 
          log_output=True, 
          save_csv=True, 
          max_depth=20,
          tau=0.9,
          reward_scale_param=9,
          gamma=0.9999):

  level_name = get_name_from_rules(rules)
  substrate_name = f'rule_obeying_harvest_{level_name}'
  num_bots = len(roles)
  # num_focal_bots = num_bots - roles.count("learner")

  config = {'substrate': substrate_name,
            'roles': roles}

  env = substrate.build(config['substrate'], roles=config['roles'], env_seed=env_seed)

  obeyed_prohibitions, obeyed_obligations = split_rules(rules)

  bots = []
  role_str = ''
  for i in range(len(roles)):
    role = config['roles'][i]
    """if i < num_focal_bots:"""
    bots.append(RuleAdjustingPolicy(env=env, 
                                    player_idx=i,
                                    log_output=log_output,
                                    look=ROLE_SPRITE_DICT[role],
                                    role=role, 
                                    num_players=num_bots,
                                    potential_obligations=POTENTIAL_OBLIGATIONS,
                                    potential_prohibitions=POTENTIAL_PROHIBITIONS,
                                    active_prohibitions=DEFAULT_PROHIBITIONS,
                                    active_obligations=DEFAULT_OBLIGATIONS,
                                    selection_mode="threshold",
                                    max_depth=max_depth,
                                    tau=tau,
                                    reward_scale_param=reward_scale_param,
                                    gamma=gamma
                                    ))
    """else:
    bots.append(RuleLearningPolicy(env=env, 
                                    look=ROLE_SPRITE_DICT[role],
                                    role=role, 
                                    player_idx=i,
                                    # other_player_looks=other_player_looks,
                                    num_focal_bots = num_focal_bots,
                                    log_output=log_output,
                                    selection_mode="threshold"
                                    ))"""
      
  for role in set(roles):
    role_str += role # video name
    role_str += str(roles.count(role))
    role_str += '_'

  timestep = env.reset()
  cum_reward = [0] * num_bots
  dead_apple_ratio = 0.0

  # actions = {key: [0] for key in range(len(bots))}
  actions = [0] * len(bots)
  # make headline of output dict#
  ROLE_LIST = ['free', 'cleaner', 'farmer', 'learner']
  ACTION_ROLE_LIST = [key + "_action" for key in config['roles']]
  data_dict = {
    (key.make_str_repr() if hasattr(key, 'make_str_repr') else key): [] 
    for key in ROLE_LIST + ACTION_ROLE_LIST + ['DESSICATED'] + \
      POTENTIAL_PROHIBITIONS + POTENTIAL_OBLIGATIONS
  }
  cur_beliefs = [] * len(POTENTIAL_PROHIBITIONS + POTENTIAL_OBLIGATIONS)

  # Configure the pygame display
  scale = 4
  fps = 5

  pygame.init()
  start_time = time.time()  # Start the timer
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))
  
  for bot in bots:
    filename = f'examples/results/policies/{bot.role}_policies.csv'
    if os.path.exists(filename):
      bot.V = read_from_csv(filename)

  for k in range(episodes):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface,
                                  (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    clock.tick(fps)

    example_bot = bots[0]
    timestep_list = [example_bot.add_non_physical_info(timestep, actions, i) for i in range(len(bots))]

    for i, bot in enumerate(bots):            
      # cum_reward[i] += timestep_bot.reward
      bot.append_to_history(timestep_list)
      if len(bot.history) > 1:
        bot.update_beliefs(actions)
      bot.obligations, bot.prohibitions = bot.threshold_rules()
        
      actions[i] = bot.step()
      # dead_apple_ratio = timestep_bot.observation['DEAD_APPLE_RATIO'] # same for every player
            
    if log_output:
      print('Actions: ' + str(actions))

    timestep = env.step(actions)
    # actions = update(actions)

    data_dict = append_to_dict(data_dict, 
                               timestep.reward, 
                               cur_beliefs, 
                               roles, 
                               actions,
                               dead_apple_ratio)

    # Saving files in superdircetory
    filename = '../videos/screen_%04d.png' % (k)
    pygame.image.save(game_display, filename)

  name = f'vers{num_iteration}_{role_str}'[:-1]
  filename = 'videos/evals/' + name + '.mov'

  if create_video:
    make_video(filename)

  if save_csv:
    for i, bot in enumerate(bots):
      filename = f'examples/results/policies/{bot.role}_policies.csv'
      save_to_csv(filename, bot.V)

  settings = get_settings(bots=bots, rules=rules)

  end_time = time.time()  # End the timer

  # return end_time - start_time, data_dict
  return settings, data_dict

  """ Profiler Run:
  ~ python3 -m cProfile -o output.prof examples/evals/evals.py 
  snakeviz output.prof 
  """

# Function to save the bot.V nested dictionaries to a CSV file
def save_to_csv(filename, data):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['Goal', 'Hashkey'] + [f'Value {i + 1}' for i in range(12)])

    # Write the data rows
    for goal, hash_dict in data.items():
      for hashkey, values in hash_dict.items():
        row_data = [str(goal), str(hashkey)] + [str(val) for val in values]
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
      if goal not in data:
          data[goal] = {}
      data[goal][hashkey] = values

  return data

def append_to_dict(data_dict: dict, reward_arr, beliefs, all_roles, actions, dead_apple_ratio) -> dict:
  for i, key in enumerate(data_dict):
    if i < 4: # player rewards
      if key in all_roles: # key 0-3 are the name of the role
        j = get_index(key, all_roles, skip_first=False)
        data_dict[key].append(reward_arr[j].item())
        """if key == 'free' and all_roles.count(key) == 2:
          j1 = get_index(key, all_roles, skip_first=True)
          data_dict['learner'].append(reward_arr[j1].item())   """       
      else: 
        data_dict[key].append(0)

    elif i < 8: # player actions
      role = key.replace('_action', '')
      if role in all_roles:
        k = get_index(role, all_roles, skip_first=False)
        data_dict[key].append(actions[k])
      else: data_dict[key].append('')

    elif i == 8:
      data_dict[key].append(dead_apple_ratio)

    else: # beliefs
      if len(beliefs) > i-9: # check if there are learner beliefs
        data_dict[key].append(beliefs[i-9]) # get beliefs (start at indec 0)
      else: data_dict[key].append(0)

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
    os.system('ffmpeg -r 20 -f image2'
              + ' -s 400x400'
              + ' -i ../videos/screen_%04d.png'
              + ' -vcodec libx264 ' 
              + ' -y '
              + filename)

# delete first row of the array
def update(actions):
  for key in actions:
    actions[key] = actions[key][1:] if len(actions[key]) > 1 else []
  return actions

if __name__ == "__main__":
  roles = ("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 0
  episodes = 200
  num_iteration = 10

  settings, data_dict = main(roles=roles,
                            rules=DEFAULT_RULES,
                            env_seed=1,
                            episodes=episodes,
                            num_iteration=1,
                            create_video=True,
                            log_output=True,
                            save_csv=False,
                            max_depth=20,
                            tau=0.5,
                            reward_scale_param=1,
                            gamma=0.8)
      
  print(sum(data_dict['cleaner']))
  print(sum(data_dict['farmer']))
  print(sum(data_dict['free']))
  print(sum(data_dict['learner']))
