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
"""A simple human player for testing `rule_obeying_harvest`.

`WASD` keys to move the character around.
`Q and E` to turn the character.
`SPACE` to fire the zapper.
`TAB` to switch between players.
`NUMBER 1` to clean the river from pollution.
`NUMBER 2` to claim resources.
`NUMBER 3` to eat from inventory
`NUMBER 4` to pay poorest player

"""

import argparse
import json
from ml_collections import config_dict

from meltingpot.python.configs.substrates import rule_obeying_harvest__complete
from meltingpot.python.configs.substrates import rule_obeying_harvest__harvest
from meltingpot.python.configs.substrates import rule_obeying_harvest__pollution
from meltingpot.python.configs.substrates import rule_obeying_harvest__territory
from meltingpot.python.human_players import level_playing_utils

environment_configs = {
    'rule_obeying_harvest__complete': rule_obeying_harvest__complete,
    'rule_obeying_harvest__harvest': rule_obeying_harvest__harvest,
    'rule_obeying_harvest__pollution': rule_obeying_harvest__pollution,
    'rule_obeying_harvest__territory': rule_obeying_harvest__territory
}

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'fireZap': level_playing_utils.get_space_key_pressed,
    'fireClean': level_playing_utils.get_key_number_one_pressed,
    'fireClaim': level_playing_utils.get_key_number_two_pressed,
    'eat': level_playing_utils.get_key_number_three_pressed,
    'pay': level_playing_utils.get_key_number_four_pressed,
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
      '--level_name', type=str, default='complete',
      help="Substrate name to load. Choose for a reduced version of the "
      "rule_obeying_harvest template: 'harvest' for only harvest, 'pollution' "
      "for harvest + pollution, or 'territory' for harvest + territory "
      "dimensions. Default is the complete environment.")
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
  level_name = f'rule_obeying_harvest__{args.level_name}'
  env_module = environment_configs[level_name]
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