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
"""Configs for substrates."""

from collections.abc import Mapping, Sequence, Set
import dataclasses
import functools
import importlib
from typing import Any

from ml_collections import config_dict


def _validated(build):
  """And adds validation checks to build function."""

  def lab2d_settings_builder(
      *,
      config: config_dict.ConfigDict,
      roles: Sequence[str],
  ) -> Mapping[str, Any]:
    """Builds the lab2d settings for the specified config and roles.

    Args:
      config: the meltingpot substrate config.
      roles: the role for each corresponding player.

    Returns:
      The lab2d settings for the substrate.
    """
    invalid_roles = set(roles) - config.valid_roles
    if invalid_roles:
      raise ValueError(f'Invalid roles: {invalid_roles!r}. Must be one of '
                       f'{config.valid_roles!r}')
    return build(config=config, roles=roles)

  return lab2d_settings_builder


def get_config(substrate: str) -> config_dict.ConfigDict:
  """Returns the specified config.

  Args:
    substrate: the name of the substrate. Must be in SUBSTRATES.

  Raises:
    ModuleNotFoundError: the config does not exist.
  """
  if substrate not in SUBSTRATES:
    raise ValueError(f'{substrate} not in {SUBSTRATES}.')
  path = f'{__name__}.{substrate}'
  module = importlib.import_module(path)
  config = module.get_config()
  with config.unlocked():
    config.lab2d_settings_builder = _validated(module.build)
  return config.lock()


SUBSTRATES: Set[str] = frozenset({
    # keep-sorted start
    'rule_obeying_harvest__apples',
    'rule_obeying_harvest__apples_cleaning',
    'rule_obeying_harvest__apples_territory',
    'rule_obeying_harvest__cleaning',
    'rule_obeying_harvest__cleaning_territory',
    'rule_obeying_harvest__complete',
    'rule_obeying_harvest__empty',
    'rule_obeying_harvest__territory',
    # keep-sorted end
})