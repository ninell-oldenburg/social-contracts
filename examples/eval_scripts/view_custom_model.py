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

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from meltingpot.python.utils.policies.lambda_rules import CLEANING_RULES, PICK_APPLE_RULES, TERRITORY_RULES
from meltingpot.python.configs.substrates.rule_obeying_harvest__complete import ROLE_SPRITE_DICT

# from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.rule_adjusting_policy import RuleAdjustingPolicy
# from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS

def main(roles, 
          episodes, 
          num_iteration, 
          rules, 
          env_seed, 
          create_video=True, 
          log_output=True, 
          log_weights=False,
          save_csv=True,
          plot_q_vals=False,
          gamma=0.999,
          tau=1.5,
          passive_learning=True,
          stochastic_act_selection=False,
          ):

  level_name = get_name_from_rules(rules)
  substrate_name = f'rule_obeying_harvest_{level_name}'
  num_bots = len(roles)
  num_focal_bots = num_bots - roles.count("learner")

  config = {'substrate': substrate_name,
            'roles': roles}

  env = substrate.build(config['substrate'], roles=config['roles'], env_seed=env_seed)

  # obeyed_prohibitions, obeyed_obligations = split_rules(rules)

  bots = []
  role_str = ''
  for i, role in enumerate(roles):
    if not role == 'learner':
      bots.append(RuleAdjustingPolicy(env=env, 
                                    player_idx=i,
                                    log_output=log_output,
                                    log_rule_prob_output=False,
                                    log_weights=log_weights,
                                    look=ROLE_SPRITE_DICT[role],
                                    role=role, 
                                    num_players=num_bots,
                                    potential_prohibitions=DEFAULT_PROHIBITIONS,
                                    potential_obligations=DEFAULT_OBLIGATIONS,
                                    active_prohibitions=DEFAULT_PROHIBITIONS,
                                    active_obligations=DEFAULT_OBLIGATIONS,
                                    gamma=gamma,
                                    tau=tau,
                                    is_learner=False,
                                    stochastic_act_selection=stochastic_act_selection,
                                    ))
    else:
      bots.append(RuleAdjustingPolicy(env=env, 
                                    player_idx=i,
                                    log_output=log_output,
                                    log_rule_prob_output=True,
                                    log_weights=log_weights,
                                    look=ROLE_SPRITE_DICT[role],
                                    role=role, 
                                    num_players=num_bots,
                                    potential_obligations=POTENTIAL_OBLIGATIONS,
                                    potential_prohibitions=POTENTIAL_PROHIBITIONS,
                                    active_prohibitions=[],
                                    active_obligations=[],
                                    gamma=gamma,
                                    tau=tau,
                                    is_learner=True,
                                    stochastic_act_selection=stochastic_act_selection,
                                    ))
      
    for bot in bots:
      bot.set_all_bots(bots)
      
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
  # start_time = time.time()  # Start the timer
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

    timestep_list = [bot.add_non_physical_info(timestep, actions, i) for i, bot in enumerate(bots)]
    last_actions = actions

    for i, bot in enumerate(bots):            
      # cum_reward[i] += timestep_bot.reward
      bot.append_to_history(timestep_list)
      actions[i] = bot.step()

    for i, bot in enumerate(bots):
      if len(bot.history) > 1:
        if passive_learning:
          if bot.role == 'learner':
            bot.update_beliefs(last_actions)
            bot.obligations, bot.prohibitions = bot.threshold_rules()
        else:
          bot.update_beliefs(last_actions)
          bot.obligations, bot.prohibitions = bot.threshold_rules()
    
    dead_apple_ratio = bots[-1].history[-1][len(bots)-1].observation['DEAD_APPLE_RATIO'] # same for every player
    cur_beliefs = bots[-1].rule_beliefs
            
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

  if plot_q_vals:
    states = get_most_frequent_states(bots[0].q_value_log)
    for state in states:
      plt.plot(bots[0].q_value_log[state])
      plt.xlabel('Episode')
      plt.ylabel('Q-value')
      plt.title(f'Q-value Evolution for State {state}')
      plt.show()

  if create_video:
    make_video(filename)

  if save_csv:
    for i, bot in enumerate(bots):
      filename = f'examples/results/policies/{bot.role}_policies.csv'
      save_to_csv(filename, bot.V)

  settings = get_settings(bots=bots, rules=rules)

  key_with_max_value = max(bots[0].hash_count, key=bots[0].hash_count.get)
  print(f"The key with the highest value is: {key_with_max_value}")
  print(f"Its value is: {bots[0].hash_count[key_with_max_value]}")

  # end_time = time.time()  # End the timer

  # return end_time - start_time, data_dict
  return settings, data_dict

  """ Profiler Run:
  ~ python3 -m cProfile -o output.prof examples/evals/evals.py 
  snakeviz output.prof 
  """
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
        if key == 'free' and all_roles.count(key) == 2:
          j1 = get_index(key, all_roles, skip_first=True)
          data_dict['learner'].append(reward_arr[j1].item())    
        else:
          data_dict[key].append(reward_arr[j].item())
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


if __name__ == "__main__":
  roles = ("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 0
  episodes = 200
  # Possible values for tau and gamma you want to test
  # taus = [0.0, 0.1, 0.5, 1.0, 1.5]
  # gammas = [0.9, 0.99, 0.999, 0.9999, 0.99999]
  num_runs = 1
  
  # Initialize results
  # results = {(gamma, tau): {'cleaner': 0, 'farmer': 0, 'free': 0} for gamma in gammas for tau in taus}
  
  for run in range(num_runs):
      """ print(f"Run number: {run+1}")
      for gamma in gammas:
        for tau in taus:"""
    
      settings, data_dict = main(roles=roles,
                                rules=DEFAULT_RULES,
                                env_seed=1,
                                episodes=episodes,
                                num_iteration=1,
                                create_video=True,
                                log_output=True,
                                log_weights=False,
                                save_csv=False,
                                plot_q_vals=False,
                                gamma=0.999,
                                tau=1.0,
                                passive_learning=True,
                                stochastic_act_selection=False,
                                )

      """results[(gamma, tau)]['cleaner'] += sum(data_dict['cleaner'])
      results[(gamma, tau)]['farmer'] += sum(data_dict['farmer'])
      results[(gamma, tau)]['free'] += sum(data_dict['free'])

  # Calculate averages
  for (gamma, tau), scores in results.items():
      scores['cleaner'] /= num_runs
      scores['farmer'] /= num_runs
      scores['free'] /= num_runs
  
  # Print or process averaged results as desired
  for (gamma, tau), scores in results.items():
      print(f"For gamma={gamma}:")
      print(f"cleaner: {scores['cleaner']:.2f}, farmer: {scores['farmer']:.2f}, free: {scores['free']:.2f}")
      print()"""

  print(sum(data_dict['cleaner']))
  print(sum(data_dict['farmer']))
  print(sum(data_dict['free']))
