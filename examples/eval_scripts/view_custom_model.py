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
from itertools import islice

import os

from meltingpot.python import substrate

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from meltingpot.python.utils.policies.lambda_rules import CLEANING_RULES, PICK_APPLE_RULES, TERRITORY_RULES
from meltingpot.python.configs.substrates.rule_obeying_harvest__complete import ROLE_SPRITE_DICT

from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS

def main(roles, episodes, num_iteration, rules, env_seed, create_video=True, log_output=True):

  level_name = get_name_from_rules(rules)
  substrate_name = f'rule_obeying_harvest_{level_name}'
  num_bots = len(roles)
  num_focal_bots = num_bots - roles.count("learner")

  config = {'substrate': substrate_name,
            'roles': roles}

  env = substrate.build(config['substrate'], roles=config['roles'], env_seed=env_seed)

  other_player_looks = [ROLE_SPRITE_DICT[role] for role in config['roles']]
  obeyed_prohibitions, obeyed_obligations = split_rules(rules)

  bots = []
  role_str = ''
  for i in range(len(roles)):
    role = config['roles'][i]
    if i < num_focal_bots:
      bots.append(RuleObeyingPolicy(env=env, 
                                    look=ROLE_SPRITE_DICT[role],
                                    role=role, 
                                    log_output=log_output,
                                    player_idx=i,
                                    prohibitions=obeyed_prohibitions,
                                    obligations=obeyed_obligations,
                                    ))
    else:
      bots.append(RuleLearningPolicy(env=env, 
                                    look=ROLE_SPRITE_DICT[role],
                                    role=role, 
                                    player_idx=i,
                                    other_player_looks=other_player_looks,
                                    num_focal_bots = num_focal_bots,
                                    log_output=log_output,
                                    selection_mode="threshold"
                                    ))
      
  for role in set(roles):
    role_str += role # video name
    role_str += str(roles.count(role))
    role_str += '_'

  timestep = env.reset()
  cum_reward = [0] * num_bots
  dead_apple_ratio = 0.0

  actions = {key: [] for key in range(len(bots))}
  # make headline of output dict
  ACTION_ROLE_LIST = [key+ "_action" for key in ROLE_SPRITE_DICT.keys()]
  data_dict = {
    (key.make_str_repr() if hasattr(key, 'make_str_repr') else key): [] 
    for key in list(ROLE_SPRITE_DICT.keys()) + \
      ACTION_ROLE_LIST + ['DESSICATED'] + POTENTIAL_PROHIBITIONS + POTENTIAL_OBLIGATIONS
  }
  cur_beliefs = [] * len(POTENTIAL_PROHIBITIONS + POTENTIAL_OBLIGATIONS)

  # Configure the pygame display
  scale = 4
  fps = 5

  pygame.init()
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))

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

    for i, bot in enumerate(bots):
      timestep_bot = dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward[i],
            discount=timestep.discount,
            observation=timestep.observation[i])
      
      cum_reward[i] += timestep_bot.reward
      bot.update_and_append_history(timestep_bot)

      if i >= num_focal_bots: # update beliefs if focal bot
        bot.update_and_append_others_history(timestep)
        if len(bot.others_history) > 2:
          other_acts = [action[0] for _, action in islice(
            actions.items(), num_focal_bots)]
          bot.update_beliefs(other_acts)
          bot.obligations, bot.prohibitions = bot.threshold_rules(threshold=0.75)
          
        cur_beliefs = bot.rule_beliefs
        
      if len(actions[i]) == 0: # action pipeline empty
        actions[i] = bot.step(timestep_bot)
        
      dead_apple_ratio = timestep_bot.observation['DEAD_APPLE_RATIO'] # same for every player
            
    if log_output:
      print(actions)

    action_list = [int(item[0]) for item in actions.values()]
    timestep = env.step(action_list)
    actions = update(actions)

    data_dict = append_to_dict(data_dict, 
                               timestep.reward, 
                               cur_beliefs, 
                               roles, 
                               action_list,
                               dead_apple_ratio)

    # Saving files in superdircetory
    filename = '../videos/screen_%04d.png' % (k)
    pygame.image.save(game_display, filename)

  name = f'vers{num_iteration}_{role_str}'[:-1]
  filename = 'videos/evals/' + name + '.mov'

  if create_video:
    make_video(filename)

  settings = get_settings(bots=bots, rules=rules)

  return settings, data_dict

  """ Profiler Run:
  ~ python3 -m cProfile -o run1.prof -s cumtime  examples/evals/evals.py """

def append_to_dict(data_dict: dict, reward_arr, beliefs, all_roles, actions, dead_apple_ratio) -> dict:
  for i, key in enumerate(data_dict):
    if i < 4: # player rewards
      if key in all_roles: # key 0-3 are the name of the role
        j = get_index(key, all_roles)
        data_dict[key].append(reward_arr[j].item())
      else: data_dict[key].append(0)

    elif i < 8: # player actions
      role = key.replace('_action', '')
      if role in all_roles:
        k = get_index(role, all_roles)
        data_dict[key].append(actions[k])
      else: data_dict[key].append('')

    elif i == 8:
      data_dict[key].append(dead_apple_ratio)

    else: # beliefs
      if len(beliefs) > i-9: # check if there are learner beliefs
        data_dict[key].append(beliefs[i-9]) # get beliefs (start at indec 0)
      else: data_dict[key].append(0)

  return data_dict

def get_index(role, all_roles):
  for i, name in enumerate(all_roles):
    if name == role:
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
  roles = ("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 1 + ('learner',) * 1
  episodes = 200
  num_iteration = 1
  setting, data_dict = main(roles=roles,
                            rules=[],
                            env_seed=1,
                            episodes=episodes, 
                            num_iteration=num_iteration, 
                            create_video=True,
                            log_output=False)
  
  print(sum(data_dict['cleaner']))
  print(sum(data_dict['farmer']))
  print(sum(data_dict['free']))
  print(sum(data_dict['learner']))