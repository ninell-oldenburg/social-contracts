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
"""Configuration for Commons Harvest with Territory.

Example video: TBD

Apples are spread around the map and can be consumed for a reward of 1. Apples
that have been consumed regrow with a per-step probability that depends on the
number of uneaten apples in a `L2` norm neighborhood of radius 2 (by default).
After an apple has been eaten and thus removed, its regrowth probability depends
on the number of uneaten apples still in its local neighborhood. With standard
parameters, it the grown rate decreases as the number of uneaten apples in the
neighborhood decreases and when there are zero uneaten apples in the
neighborhood then the regrowth rate is zero. As a consequence, a patch of apples
that collectively doesn't have any nearby apples, can be irrevocably lost if all
apples in the patch are consumed. Therefore, agents must exercise restraint when
consuming apples within a patch. Notice that in a single agent situation, there
is no incentive to collect the last apple in a patch (except near the end of the
episode). However, in a multi-agent situation, there is an incentive for any
agent to consume the last apple rather than risk another agent consuming it.
This creates a tragedy of the commons from which the substrate derives its name.

This mechanism was first described in Janssen et al (2010) and adapted for
multi-agent reinforcement learning in Perolat et al (2017).

Janssen, M.A., Holahan, R., Lee, A. and Ostrom, E., 2010. Lab experiments for
the study of social-ecological systems. Science, 328(5978), pp.613-617.

Perolat, J., Leibo, J.Z., Zambaldi, V., Beattie, C., Tuyls, K. and Graepel, T.,
2017. A multi-agent reinforcement learning model of common-pool
resource appropriation. In Proceedings of the 31st International Conference on
Neural Information Processing Systems (pp. 3646-3655).

Clean Up was first described in Hughes et al. (2018).

Hughes, E., Leibo, J.Z., Phillips, M., Tuyls, K., Duenez-Guzman, E.,
Castaneda, A.G., Dunning, I., Zhu, T., McKee, K., Koster, R. and Roff, H., 2018,
Inequity aversion improves cooperation in intertemporal social dilemmas. In
Proceedings of the 32nd International Conference on Neural Information
Processing Systems (pp. 3330-3340).

Territory was first described by Leibo et al. (2021)

Leibo, J. Z., Due??ez-Guzman, E. A., Vezhnevets, A., Agapiou, J. P., Sunehag, P., 
Koster, R., ... & Graepel, T. (2021, July). Scalable evaluation of multi-agent 
reinforcement learning with melting pot. In International Conference on 
Machine Learning (pp. 6187-6199). PMLR.
"""

from typing import Any, Dict, Mapping, Sequence

from ml_collections import config_dict
import numpy as np

from meltingpot.python.utils.substrates import colors
from meltingpot.python.utils.substrates import game_object_utils
from meltingpot.python.utils.substrates import shapes
from meltingpot.python.utils.substrates import specs

PrefabConfig = game_object_utils.PrefabConfig

APPLE_RESPAWN_RADIUS = 2.0
REGROWTH_PROBABILITIES = [0.0, 0.0025, 0.005, 0.025]

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
WAAA____A_____________A____AAW
WAA____AAA___________AAA____AW
WA____AAAAA_________AAAAA____W
W______AAA___________AAA_____W
W_______A_____________A______W
W__A___________A__________A__W
W_AAA__Q______AAA____Q___AAA_W
WAAAAA_______AAAAA______AAAAAW
W_AAA_________AAA________AAA_W
W__A___________A__________A__W
W__GGGGGGGGGGGGGGGGGGGGGGGG__W
W__GGGGGGGGGGGGGGGGGGGGGGGG__W
WGGGGGGGGGGGGGGGGGGGGGGGGGGGGW
WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
"""

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "W": "wall",
    "S": "river",
    "_": "grass",
    "Q": {"type": "all", "list": ["grass", "inside_spawn_point"]},
    "s": {"type": "all", "list": ["grass", "shadow_n"]},
    "A": {"type": "all", "list": ["grass", "apple"]},
    "G": {"type": "all", "list": ["grass", "spawn_point"]},
}

_COMPASS = ["N", "E", "S", "W"]

MARKING_SPRITE = """
oxxxxxxo
xoxxxxox
xxoxxoxx
xxxooxxx
xxxooxxx
xxoxxoxx
xoxxxxox
oxxxxxxo
"""

def get_marking_palette(alpha: float) -> Dict[str, Sequence[int]]:
  alpha_uint8 = int(alpha * 255)
  assert alpha_uint8 >= 0.0 and alpha_uint8 <= 255, "Color value out of range."
  return {"x": shapes.ALPHA, "o": (0, 0, 0, alpha_uint8)}

WALL = {
    "name": "wall",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "wall",
                "stateConfigs": [{
                    "state": "wall",
                    "layer": "upperPhysical",
                    "sprite": "Wall",
                }],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Wall"],
                "spriteShapes": [shapes.WALL],
                "palettes": [{"*": (95, 95, 95, 255),
                              "&": (100, 100, 100, 255),
                              "@": (109, 109, 109, 255),
                              "#": (152, 152, 152, 255)}],
                "noRotates": [False]
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "zapHit"
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "cleanHit"
            }
        },
    ]
}

GRASS = {
    "name":
        "grass",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState":
                    "grass",
                "stateConfigs": [
                    {
                        "state": "grass",
                        "layer": "background",
                        "sprite": "Grass"
                    },
                    {
                        "state": "dessicated",
                        "layer": "background",
                        "sprite": "Floor"
                    },
                ],
            }
        },
        {
            "component": "Transform",
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Grass", "Floor"],
                "spriteShapes": [
                    shapes.GRASS_STRAIGHT, shapes.GRAINY_FLOOR
                ],
                "palettes": [{
                    "*": (158, 194, 101, 255),
                    "@": (170, 207, 112, 255)
                }, {
                    "*": (220, 205, 185, 255),
                    "+": (210, 195, 175, 255),
                }],
                "noRotates": [False, False]
            }
        },
    ]
}

SPAWN_POINT = {
    "name": "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "alternateLogic",
                    "groups": ["spawnPoints"]
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

INSIDE_SPAWN_POINT = {
    "name": "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "alternateLogic",
                    "groups": ["insideSpawnPoints"]
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

# Primitive action components.
# pylint: disable=bad-whitespace
# pyformat: disable
NOOP        = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
FORWARD     = {"move": 1, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
STEP_RIGHT  = {"move": 2, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
BACKWARD    = {"move": 3, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
STEP_LEFT   = {"move": 4, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
TURN_LEFT   = {"move": 0, "turn": -1, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
TURN_RIGHT  = {"move": 0, "turn":  1, "fireZap": 0, "fireClean": 0, "fireClaim": 0}
FIRE_ZAP    = {"move": 0, "turn":  0, "fireZap": 1, "fireClean": 0, "fireClaim": 0}
FIRE_CLEAN  = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 1, "fireClaim": 0}
FIRE_CLAIM  = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 1}
# pyformat: enable
# pylint: enable=bad-whitespace

ACTION_SET = (
    NOOP,
    FORWARD,
    BACKWARD,
    STEP_LEFT,
    STEP_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    FIRE_ZAP,
    FIRE_CLEAN,
    FIRE_CLAIM,
)

TARGET_SPRITE_SELF = {
    "name": "Self",
    "shape": shapes.CUTE_AVATAR,
    "palette": shapes.get_palette((50, 100, 200)),
    "noRotate": True,
}

def get_brush_palette(
    base_color: shapes.Color) -> Mapping[str, shapes.ColorRGBA]:
  return {
      "*": base_color + (255,),
      "&": shapes.scale_color(base_color, 0.75, 255),
      "o": shapes.scale_color(base_color, 0.55, 255),
      "O": (70, 70, 70, 255),
      "-": (143, 96, 74, 255),
      "+": (117, 79, 61, 255),
      "k": (199, 176, 135, 255),
      "x": shapes.ALPHA,
  }

def get_dry_painted_wall_palette(base_color: shapes.Color
                                 ) -> Mapping[str, shapes.ColorRGBA]:
  return {
      "*": shapes.scale_color(base_color, 0.75, 200),
      "#": shapes.scale_color(base_color, 0.90, 150),
  }

PLAYER_COLOR_PALETTES = []
BRUSH_PALETTES = []
for human_readable_color in colors.human_readable:
  PLAYER_COLOR_PALETTES.append(shapes.get_palette(human_readable_color))
  BRUSH_PALETTES.append(get_brush_palette(human_readable_color))

def create_resource(num_players: int) -> PrefabConfig:
  """Configure the prefab to use for all resource objects."""
  # Setup unique states corresponding to each player who can claim the resource.
  claim_state_configs = []
  claim_sprite_names = []
  claim_sprite_rgb_colors = []
  for player_idx in range(num_players):
    lua_player_idx = player_idx + 1
    player_color = colors.human_readable[player_idx]
    wet_sprite_name = "Color" + str(lua_player_idx) + "ResourceSprite"
    claim_state_configs.append({
        "state": "claimed_by_" + str(lua_player_idx),
        "layer": "upperPhysical",
        "sprite": wet_sprite_name,
        "groups": ["claimedResources"]
    })
    claim_sprite_names.append(wet_sprite_name)
    # Use alpha channel to make transparent version of claiming agent's color.
    wet_paint_color = player_color + (75,)
    claim_sprite_rgb_colors.append(wet_paint_color)

  prefab = {
      "name": "resource",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "unclaimed",
                  "stateConfigs": [
                      {"state": "unclaimed",
                       "layer": "upperPhysical",
                       "sprite": "UnclaimedResourceSprite"},
                      {"state": "destroyed"},
                  ] + claim_state_configs,
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "colored_square",
                  "spriteNames": claim_sprite_names,
                  "spriteRGBColors": claim_sprite_rgb_colors
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Resource",
              "kwargs": {
                  "initialHealth": 2,
                  "destroyedState": "destroyed",
                  "reward": 1.0,
                  "rewardRate": 0.01,
                  "rewardDelay": 25,
                  "delayTillSelfRepair": 15,
                  "selfRepairProbability": 0.1,
              }
          },
      ]
  }
  return prefab

def create_resource_texture() -> PrefabConfig:
  """Configure the background texture for a resource. It looks like a wall."""
  prefab = {
      "name": "resource_texture",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "unclaimed",
                  "stateConfigs": [
                      {"state": "unclaimed",
                       "layer": "lowerPhysical",
                       "sprite": "UnclaimedResourceSprite"},
                      {"state": "destroyed"},
                  ],
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["UnclaimedResourceSprite",],
                  "spriteShapes": [shapes.WALL],
                  "palettes": [{"*": (61, 61, 61, 255),
                                "#": (80, 80, 80, 255)}],
                  "noRotates": [True]
              }
          },
          {
              "component": "Transform",
          },
      ]
  }
  return prefab


def create_reward_indicator(num_players) -> PrefabConfig:
  """Configure object indicating if a resource is currently providing reward."""
  # Setup unique states corresponding to each player who can claim the resource.
  claim_state_configs = []
  claim_sprite_names = []
  claim_sprite_shapes = []
  claim_palettes = []
  claim_no_rotates = []
  for player_idx in range(num_players):
    lua_player_idx = player_idx + 1
    player_color = colors.human_readable[player_idx]
    dry_sprite_name = "Color" + str(lua_player_idx) + "DryPaintSprite"
    claim_state_configs.append({
        "state": "dry_claimed_by_" + str(lua_player_idx),
        "layer": "overlay",
        "sprite": dry_sprite_name,
    })
    claim_sprite_names.append(dry_sprite_name)
    claim_sprite_shapes.append(shapes.WALL)
    claim_palettes.append(get_dry_painted_wall_palette(player_color))
    claim_no_rotates.append(True)
  prefab = {
      "name": "reward_indicator",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "inactive",
                  "stateConfigs": [
                      {"state": "inactive"},
                  ] + claim_state_configs,
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": claim_sprite_names,
                  "spriteShapes": claim_sprite_shapes,
                  "palettes": claim_palettes,
                  "noRotates": claim_no_rotates
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "RewardIndicator",
          },
      ]
  }
  return prefab

def create_apple_prefab(regrowth_radius=-1.0,  # pylint: disable=dangerous-default-value
                        regrowth_probabilities=[0, 0.0, 0.0, 0.0]):
  """Creates the apple prefab with the provided settings."""
  growth_rate_states = [
      {
          "state": "apple",
          "layer": "lowerPhysical",
          "sprite": "Apple",
          "groups": ["apples"]
      },
      {
          "state": "appleWait",
          "layer": "logic",
          "sprite": "AppleWait",
      },
  ]
  # Enumerate all possible states for a potential apple. There is one state for
  # each regrowth rate i.e., number of nearby apples.
  upper_bound_possible_neighbors = np.floor(np.pi*regrowth_radius**2+1)+1
  for i in range(int(upper_bound_possible_neighbors)):
    growth_rate_states.append(dict(state="appleWait_{}".format(i),
                                   layer="logic",
                                   groups=["waits_{}".format(i)],
                                   sprite="AppleWait"))

  apple_prefab = {
      "name": "apple",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "apple",
                  "stateConfigs": growth_rate_states,
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["Apple", "AppleWait"],
                  "spriteShapes": [shapes.APPLE, shapes.FILL],
                  "palettes": [
                      {"x": (0, 0, 0, 0),
                       "*": (214, 88, 88, 255),
                       "#": (194, 79, 79, 255),
                       "o": (53, 132, 49, 255),
                       "|": (102, 51, 61, 255)},
                      {"i": (0, 0, 0, 0)}],
                  "noRotates": [True, True]
              }
          },
          {
              "component": "Edible",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "rewardForEating": 1.0,
              }
          },
          {
              "component": "DensityRegrow",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "radius": regrowth_radius,
                  "regrowthProbabilities": regrowth_probabilities,
                  "maxAppleGrowthRate": 0.9,
                  "thresholdDepletion": 0.01,
                  "thresholdRestoration": 0.0,
              }
          },
      ]
  }

  return apple_prefab


def create_prefabs(regrowth_radius=-1.0,
                   # pylint: disable=dangerous-default-value
                   regrowth_probabilities=[0, 0.0, 0.0, 0.0],
                   num_players=7) -> PrefabConfig:
  """Returns a dictionary mapping names to template game objects."""
  prefabs = {
      "wall": WALL,
      "grass": GRASS,
      "spawn_point": SPAWN_POINT,
      "inside_spawn_point": INSIDE_SPAWN_POINT,
      "resource_texture": create_resource_texture(),
      "reward_indicator": create_reward_indicator(num_players),
      "resource": create_resource(num_players=num_players),
  }
  prefabs["apple"] = create_apple_prefab(
      regrowth_radius=regrowth_radius,
      regrowth_probabilities=regrowth_probabilities)
  return prefabs


def create_scene():
  # Create the scene object, a non-physical object to hold global logic.
  scene = {
      "name": "scene",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "scene",
                  "stateConfigs": [{
                      "state": "scene",
                  }],
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "GlobalData",
          },
          {
              "component": "Neighborhoods",
              "kwargs": {}
          },
          {
              "component": "RiverMonitor",
              "kwargs": {},
          },
          {
              "component": "StochasticIntervalEpisodeEnding",
              "kwargs": {
                  "minimumFramesPerEpisode": 1000,
                  "intervalLength": 100,  # Set equal to unroll length.
                  "probabilityTerminationPerInterval": 0.2
              }
          },
      ]
  }
  return scene

def create_avatar_object(player_idx: int,
                         target_sprite_self: Dict[str, Any],
                         spawn_group: str) -> Dict[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)
  custom_sprite_map = {source_sprite_self: target_sprite_self["name"]}

  color_palette = PLAYER_COLOR_PALETTES[player_idx]
  paintbrush_palette = BRUSH_PALETTES[player_idx]

  live_state_name = "player{}".format(lua_index)

  avatar_object = {
      "name": "avatar",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": live_state_name,
                  "stateConfigs": [
                      # Initial player state.
                        {
                        "state": live_state_name,
                        "layer": "upperPhysical",
                        "sprite": source_sprite_self,
                        "contact": "avatar",
                        "groups": ["players"]
                        },
                      # Player wait state used when they have been zapped out
                        {
                        "state": "playerWait",
                        "groups": ["playerWaits"]
                        },
                  ]
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": [source_sprite_self],
                  "spriteShapes": [shapes.CUTE_AVATAR],
                  "palettes": [shapes.get_palette(
                      colors.human_readable[player_idx])],
                  "noRotates": [True]
              }
          },
          {
              "component": "AdditionalSprites",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "customSpriteNames": [target_sprite_self["name"]],
                  "customSpriteShapes": [target_sprite_self["shape"]],
                  "customPalettes": [target_sprite_self["palette"]],
                  "customNoRotates": [target_sprite_self["noRotate"]],
              }
          },
          {
              "component": "Avatar",
              "kwargs": {
                  "index": lua_index,
                  "aliveState": live_state_name,
                  "waitState": "playerWait",
                  "speed": 1.0,
                  "spawnGroup": spawn_group,
                  "postInitialSpawnGroup": "spawnPoints",
                  "actionOrder": ["move", 
                                  "turn", 
                                  "fireZap",
                                  "fireClean",
                                  "fireClaim",
                                  ],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
                      "fireClean": {"default": 0, "min": 0, "max": 1},
                      "fireClaim": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
                  "spriteMap": custom_sprite_map,
              }
          },
          {
              "component": "AllNonselfCumulants",
          },
          {
              "component": "Paintbrush",
              "kwargs": {
                  "shape": shapes.PAINTBRUSH,
                  "palette": paintbrush_palette,
                  "playerIndex": lua_index,
              }
          },
          {
              "component": "ResourceClaimer",
              "kwargs": {
                  "color": color_palette["*"],
                  "playerIndex": lua_index,
                  "beamLength": 2,
                  "beamRadius": 1,
                  "beamWait": 0,
              }
          },
          {
              "component": "Taste",
              "kwargs": {
                  "role": "free",
                  "rewardAmount": 1,
              }
          },
          {
              "component": "Zapper",
              "kwargs": {
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
                  "framesTillRespawn": 4,
                  "penaltyForBeingZapped": 0,
                  "rewardForZapping": 0,
              }
          },
          {
              "component": "ReadyToShootObservation",
          },
          {
              "component": "LocationObserver",
              "kwargs": {
                  "objectIsAvatar": True,
                  "alsoReportOrientation": True
              }
          },
          {
              "component": "AvatarMetricReporter",
              "kwargs": {
                  "metrics": [
                      {
                          "name": "PLAYER_ATE_APPLE",
                          "type": "Doubles",
                          "shape": [],
                          "component": "Taste",
                          "variable": "player_ate_apple",
                      },
                      {
                          "name": "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP",
                          "type": "Doubles",
                          "shape": [],
                          "component": "Zapper",
                          "variable": "num_others_player_zapped_this_step",
                      },
                      {
                          "name": "NUM_OTHERS_WHO_ATE_THIS_STEP",
                          "type": "Doubles",
                          "shape": [],
                          "component": "AllNonselfCumulants",
                          "variable": "num_others_who_ate_this_step",
                      },
                  ]
              }
          },
      ]
  }
  return avatar_object

def create_marking_overlay(player_idx: int) -> Mapping[str, Any]:
  """Create a graduated sanctions marking overlay object."""
  # Lua is 1-indexed.
  lua_idx = player_idx + 1

  marking_object = {
      "name": "avatar_marking",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "avatarMarkingWait",
                  "stateConfigs": [
                      # Declare one state per level of the hit logic.
                      {"state": "level_1",
                       "layer": "superOverlay",
                       "sprite": "sprite_for_level_1"},
                      {"state": "level_2",
                       "layer": "superOverlay",
                       "sprite": "sprite_for_level_2"},

                      # Invisible inactive (zapped out) overlay type.
                      {"state": "avatarMarkingWait",
                       "groups": ["avatarMarkingWaits"]},
                  ]
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["sprite_for_level_1",
                                  "sprite_for_level_2"],
                  "spriteShapes": [MARKING_SPRITE,
                                   MARKING_SPRITE],
                  "palettes": [get_marking_palette(0.0),
                               get_marking_palette(1.0)],
                  "noRotates": [True] * 3
              }
          },
          {
              "component": "GraduatedSanctionsMarking",
              "kwargs": {
                  "playerIndex": lua_idx,
                  "waitState": "avatarMarkingWait",
                  "hitName": "zapHit",
                  "recoveryTime": 50,
                  "hitLogic": [
                      {"levelIncrement": 1,
                       "sourceReward": 0,
                       "targetReward": 0,
                       "freeze": 25},
                      {"levelIncrement": -1,
                       "sourceReward": 0,
                       "targetReward": 0,
                       "remove": True}
                  ],
              }
          },
      ]
  }
  return marking_object


def create_avatar_and_associated_objects(num_players):
  """Returns list of avatars and their associated 
  marking objects of length 'num_players'."""
  avatar_objects = []
  additional_objects = []
  for player_idx in range(0, num_players):
    spawn_group = "spawnPoints"
    if player_idx < 2:
      # The first two player slots always spawn closer to the apples.
      spawn_group = "insideSpawnPoints"

    game_object = create_avatar_object(player_idx,
                                       TARGET_SPRITE_SELF,
                                       spawn_group=spawn_group)
    avatar_objects.append(game_object)

    marking_object = create_marking_overlay(player_idx)
    additional_objects.append(marking_object)

  return avatar_objects + additional_objects


def get_config():
  """Default configuration for training on the rule_obeying_harvest level."""
  config = config_dict.ConfigDict()

  # Action set configuration.
  config.action_set = ACTION_SET
  # Observation format configuration.
  config.individual_observation_names = [
      "RGB",
      "READY_TO_SHOOT",

      # Cumulants.
      "PLAYER_ATE_APPLE",
      "PLAYER_CLEANED",
      "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP",

      # Global switching signals for puppeteers.
      "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
      "NUM_OTHERS_WHO_ATE_THIS_STEP",

      # Debug only (do not use the following observations in policies).
      "POSITION",
      "ORIENTATION",
  ]
  config.global_observation_names = [
      "WORLD.RGB",
  ]

  # The specs of the environment (from a single-agent perspective).
  config.action_spec = specs.action(len(ACTION_SET))
  config.timestep_spec = specs.timestep({
     "RGB": specs.OBSERVATION["RGB"],
      "READY_TO_SHOOT": specs.OBSERVATION["READY_TO_SHOOT"],
      # Cumulants.
      "PLAYER_ATE_APPLE": specs.float64(),
      "NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP": specs.float64(),
      # Global switching signals for puppeteers.
      "NUM_OTHERS_WHO_ATE_THIS_STEP": specs.float64(),
      # Debug only (do not use the following observations in policies).
      "POSITION": specs.OBSERVATION["POSITION"],
      "ORIENTATION": specs.OBSERVATION["ORIENTATION"],
      "WORLD.RGB": specs.rgb(168, 240),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"default"})
  config.default_player_roles = ("default",) * 1

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate definition given player roles."""
  del config
  num_players = len(roles)
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="rule_obeying_harvest",
      levelDirectory="meltingpot/lua/levels",
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=5000,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": create_avatar_and_associated_objects(num_players),
          "scene": create_scene(),
          "prefabs": create_prefabs(APPLE_RESPAWN_RADIUS,
                                    REGROWTH_PROBABILITIES,
                                    num_players),
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  )
  return substrate_definition
