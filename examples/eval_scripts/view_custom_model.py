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

import argparse

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from itertools import islice

import os

from meltingpot.python import substrate

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_W_FARMER_HAT,
   'learner': shapes.CUTE_AVATAR_W_STUDENT_HAT,
   }

def main(roles, episodes, num_iteration, create_video=True):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--substrate_name",
      type=str,
      default="complete",
      help="Substrate name to load. Choose for a reduced version of the "
      "rule_obeying_harvest template: 'harvest' for only harvest, 'pollution' "
      "for harvest + pollution, or 'territory' for harvest + territory "
      "dimensions. Default is the complete environment.")

  args = parser.parse_args()

  level_name = args.substrate_name
  substrate_name = f'rule_obeying_harvest__{level_name}'
  num_bots = len(roles)
  num_focal_bots = num_bots - roles.count("learner")

  config = {'substrate': substrate_name,
            'roles': roles}

  env = substrate.build(config['substrate'], roles=config['roles'])

  player_looks = [ROLE_SPRITE_DICT[role] for role in config['roles']]

  bots = []
  role_str = ''
  for i in range(len(roles)):
    if i < num_focal_bots:
      bots.append(RuleObeyingPolicy(env=env, 
                                    role=config['roles'][i], 
                                    player_idx=i))
    else:
      bots.append(RuleLearningPolicy(env=env, 
                                    role=config['roles'][i], 
                                    player_idx=i,
                                    player_looks=player_looks,
                                    selection_mode="threshold"))
      
  for role in set(roles):
    role_str += role # video name
    role_str += str(roles.count(role))
    role_str += '_'

  timestep = env.reset()
  cum_reward = [0] * num_bots

  actions = {key: [] for key in range(len(bots))}

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
      
      if len(actions[i]) == 0: # action pipeline empty
        if i < num_focal_bots:
          actions[i] = bot.step(timestep_bot)
        else:
          other_agents_actions = [action[0] for _, action in islice(
            actions.items(), num_focal_bots)]
          other_players_observations = [observation for observation in islice(
            timestep.observation, num_focal_bots)]
          actions[i] = bot.step(timestep_bot, 
                                other_players_observations, 
                                other_agents_actions)

      else: # action pipeline not empty
        if i >= num_focal_bots: # still update learners' beliefs
          other_agents_actions = [action[0] for _, action in islice(
            actions.items(), num_focal_bots)]
          if len(bot.others_history) >= 2:
            bot.update_beliefs(timestep_bot.observation,
                             other_agents_actions)
            
    print(actions)
    action_list = [int(item[0]) for item in actions.values()]
    timestep = env.step(action_list)
    actions = update(actions)

    # Saving files in superdircetory
    filename = '../videos/screen_%04d.png' % (k)
    pygame.image.save(game_display, filename)

  name = f'vers{num_iteration}_{role_str}'[:-1]
  filename = 'videos/evals/' + name + '.mov'

  if create_video:
    make_video(filename)

  results = create_result_dict(cum_reward=cum_reward, 
                             bots=bots, 
                             episodes=episodes)

  return results

  """ Profiler Run:
  ~ python3 -m cProfile -o run1.prof -s cumtime  examples/evals/evals.py """

def create_result_dict(cum_reward, bots, episodes):
  results = {'num_episodes': episodes,
            'free': 0,
            'cleaner': 0,
            'farmer': 0,
            'learner': 0,
            'player_rewards': cum_reward,
            'cum_reward': sum(cum_reward),
            'cum_reward_focal': 0,
            'cum_reward_learners': 0,
            'active_obligations': set(),
            'active_prohibitions': set(),
            'learned_obligations': set(),
            'learned_prohibitions': set(),
            }
  
  for i, agent in enumerate(bots):
    results[agent.role] += 1
    if isinstance(agent, RuleLearningPolicy):
      results['cum_reward_learners'] += cum_reward[i]
      for rule in agent.obligations + agent.prohibitions:
        results['learned_obligations'].update([rule.make_str_repr()])
        results['learned_prohibitions'].update([rule.make_str_repr()])

    elif isinstance(agent, RuleObeyingPolicy):
      results['cum_reward_focal'] += cum_reward[i]
      for rule in agent.obligations + agent.prohibitions:
        if hasattr(rule, 'role') and rule.role == agent.role:
          results['active_obligations'].update([rule.make_str_repr()])
        results['active_prohibitions'].update([rule.make_str_repr()])

  for key in results.keys():
    results[key] = [str(results[key])]

  return results

def make_video(filename):
    print('\nCreating video.\n')
    os.system('ffmpeg -r 20 -f image2'
              + ' -s 400x400'
              + ' -i ../videos/screen_%04d.png'
              + ' -vcodec libx264 ' 
              + filename)

# delete first row of the array
def update(actions):
  for key in actions:
    actions[key] = actions[key][1:] if len(actions[key]) > 1 else []
  return actions

if __name__ == "__main__":
  roles = ("cleaner",) * 0 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 1
  episodes = 200
  num_iteration = 1
  main(roles=roles, episodes=episodes, num_iteration=num_iteration, create_video=True)