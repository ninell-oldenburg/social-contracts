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

from meltingpot.python import substrate

from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy


def main():
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

  substrate_name = f'rule_obeying_harvest__{args.substrate_name}'
  num_bots = substrate.get_config(substrate_name).default_player_roles

  config = {'substrate': substrate_name,
            'roles': list(num_bots)}

  env = substrate.build(config['substrate'], roles=config['roles'])

  bots = [RuleObeyingPolicy(env=env) for _ in range(len(num_bots))]

  timestep = env.reset()

  actions = np.zeros((len(bots), 1), dtype=int)

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

  for _ in range(20):
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
      
      next_move = bot.step(timestep_bot)
      if actions.shape[1] < next_move.shape[0]:
        actions.resize(actions.shape[0], next_move.shape[0])
      elif actions.shape[0] == 0:
        actions.resize(len(bots), next_move.shape[0])
      actions[i] = next_move.T

    # print(actions[:][0])
    timestep = env.step(actions[:][0])
    if actions.shape[1] > 1:
      actions = np.delete(actions, 0, 0)

# delete first row of the array
def update(input_array) -> np.array:
  num_bots = input_array.shape[0]
  new_array = np.zeros((num_bots, 1), dtype=int)
  for i in range(num_bots):
    if len(input_array[i]) > 0:
      new_array[i] = input_array[i][:1]

  return new_array

if __name__ == "__main__":
  main()

