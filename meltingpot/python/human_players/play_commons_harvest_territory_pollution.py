# Copyright 2022 DeepMind Technologies Limited.
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
"""A simple human player for testing `commons_harvest_territory_pollution`.

`WASD` keys to move the character around.
`Q and E` to turn the character.
`SPACE` to fire the zapper.
`TAB` to switch between players.
`NUMBER 1` to clean the river from pollution.
`NUMBER 2` to claim resources.
`X KEY` to pickup apples.
`C KEY` to grasp apples.
`Z KEY` to hold apples.
`LEFT SHIFT` to shove apples.
`RIGHT SHIFT` to pull apples.

"""

import argparse
import json
from ml_collections import config_dict

from meltingpot.python.configs.substrates import commons_harvest_territory_pollution
from meltingpot.python.configs.substrates import commons_harvest_territory_pollution__wo_territory 
from meltingpot.python.configs.substrates import commons_harvest_territory_pollution__wo_pollution
from meltingpot.python.configs.substrates import commons_harvest_territory_pollution__wo_territory_pollution
from meltingpot.python.human_players import level_playing_utils


environment_configs = {
    'commons_harvest_territory_pollution': commons_harvest_territory_pollution,
    'commons_harvest_territory_pollution__wo_territory': commons_harvest_territory_pollution__wo_territory,
    'commons_harvest_territory_pollution__wo_pollution': commons_harvest_territory_pollution__wo_pollution,
    'commons_harvest_territory_pollution__wo_territory_pollution': commons_harvest_territory_pollution__wo_territory_pollution,
}

def get_push_pull() -> int:
  """Sets shove to either -1, 0, or 1."""
  if level_playing_utils.get_right_shift_pressed():
    return 1
  if level_playing_utils.get_left_control_pressed():
    return -1
  return 0

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'fireZap': level_playing_utils.get_space_key_pressed,
    'fireClean': level_playing_utils.get_key_number_one_pressed,
    'fireClaim': level_playing_utils.get_key_number_two_pressed,
    'pickup': level_playing_utils.get_key_x_pressed,
    'grasp': level_playing_utils.get_key_c_pressed,
    # Grappling actions
    'hold': level_playing_utils.get_key_z_pressed,
    'shove': get_push_pull,
}


def verbose_fn(env_timestep, player_index, current_player_index):
  """Print using this function once enabling the option --verbose=True."""
  lua_index = player_index + 1
  cleaned = env_timestep.observation[f'{lua_index}.PLAYER_CLEANED']
  ate = env_timestep.observation[f'{lua_index}.PLAYER_ATE_APPLE']
  num_zapped_this_step = env_timestep.observation[
      f'{lua_index}.NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP']
  num_others_cleaned = env_timestep.observation[
      f'{lua_index}.NUM_OTHERS_WHO_CLEANED_THIS_STEP']
  num_others_ate = env_timestep.observation[
      f'{lua_index}.NUM_OTHERS_WHO_ATE_THIS_STEP']
  # Only print observations from current player.
  if player_index == current_player_index:
    print(f'player: {player_index} --- player_cleaned: {cleaned} --- ' +
          f'player_ate_apple: {ate} --- num_others_cleaned: ' +
          f'{num_others_cleaned} --- num_others_ate: {num_others_ate} ' +
          f'---num_others_player_zapped_this_step: {num_zapped_this_step}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='commons_harvest_territory_pollution',
      help='Level name to load, type "commons_harvest_pollution" to disable \
        territory component, "commons_harvest_territory" to disable pollution \
        component, "commons_harvest" to disable both')
  parser.add_argument(
      '--observation', type=str, default='RGB', help='Observation to render')
  parser.add_argument(
      '--settings', type=json.loads, default={}, help='Settings as JSON string')
  # Activate verbose mode with --verbose=True.
  parser.add_argument(
      '--verbose', type=bool, default=False, help='Print debug information')
  # Activate events printing mode with --print_events=True.
  parser.add_argument(
      '--print_events', type=bool, default=False, help='Print events')

  args = parser.parse_args()
  # env_module is commons_harvest_territory_pollution.py
  env_module = environment_configs[args.level_name]
  env_config = env_module.get_config()
  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)

  level_playing_utils.run_episode(
      args.observation, args.settings, _ACTION_MAP,
      env_config, level_playing_utils.RenderType.PYGAME,
      verbose_fn=verbose_fn if args.verbose else None,
      print_events=args.print_events)


if __name__ == '__main__':
  main()
