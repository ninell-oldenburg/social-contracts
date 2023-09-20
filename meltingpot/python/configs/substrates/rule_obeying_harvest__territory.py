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
"""Configuration for Commons Harvest with Territory and Pollution.

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

Leibo, J. Z., Dueñez-Guzman, E. A., Vezhnevets, A., Agapiou, J. P., Sunehag, P., 
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
from meltingpot.python.utils.policies.rule_adjusting_policy import ROLE_SPRITE_DICT, APPLE_RESPAWN_RADIUS, REGROWTH_PROBABILITIES, OBSERVATION_RADIUS, REMOVE_HIT_PLAYER, PENALTY_FOR_BEING_ZAPPED, THRESHOLD_APPLE_DEPLETION, THRESHOLD_APPLE_RESTAURATION, DEFAULT_MAX_LIFE_SPAN, INT_TO_ROLE, ROLE_TO_INT

PrefabConfig = game_object_utils.PrefabConfig

ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWW
WFHFHFHFHFHFHFHHFHFFHFW
WFHFHFHFHFHFHFHHFHFFHFW
W========+~FHFHHFHf===W
W   P     ===+~Sf     W
W      P     <~Sf  P  W
W          P <~S>     W
WT^TAAAAA^T^T;~SAAAT^TW
W____AAA_______AAA____W
W_____A____A____A_____W
W____GGG__AAA__GGG____W
W____GGG_AAAAA_GGG____W
W____GGG__AAA__GGG____W
W_Q__A_____A_____A__Q_W
W___AAA___GGG___AAA___W
W__AAAAA___Q___AAAAA__W
W___AAA_________AAA___W
W____A____GGG____A____W
W________GGGGG________W
W_____GGG_____GGG_____W
WWWWWWWWWWWWWWWWWWWWWWW
WD-----WD-----WD------W
WWWWWWWWWWWWWWWWWWWWWWW
WD-----WD-----WD------W
WWWWWWWWWWWWWWWWWWWWWWW
"""

x_size = ASCII_MAP.find('\n', 1) -2
y_size = ASCII_MAP.count('\n') -2
MAP_SIZE = (x_size, y_size) # lua is i-indexed

# `prefab` determines which prefab game object to use for each `char` in the
# ascii map.
CHAR_PREFAB_MAP = {
    "W": "wall",
    " ": {"type": "all", "list": ["sand", "resource"]},
    "P": {"type": "all", "list": ["sand", "resource", "spawn_point"]},
    "+": {"type": "all", "list": ["sand", "resource", "shadow_e"]},
    "f": {"type": "all", "list": ["sand", "resource", "shadow_w"]},
    ";": {"type": "all", "list": ["sand", "grass_edge", "resource", "shadow_e"]},
    ",": {"type": "all", "list": ["sand", "grass_edge", "resource", "shadow_w"]},
    "^": {"type": "all", "list": ["sand", "grass_edge", "resource"]},
    "=": {"type": "all", "list": ["sand", "resource"]},
    ">": {"type": "all", "list": ["sand", "resource", "shadow_w",]},
    "<": {"type": "all", "list": ["sand", "resource", "shadow_e",]},
    "T": {"type": "all", "list": ["sand", "resource", "grass_edge"]},
    "S": "river",
    "H": {"type": "all", "list": ["river"]},
    "F": {"type": "all", "list": ["river"]},
    "~": {"type": "all", "list": ["river", "shadow_w",]},
    "_": {"type": "all", "list": ["grass", "resource"]},
    "Q": {"type": "all", "list": ["grass", "resource", "inside_spawn_point"]},
    "A": {"type": "all", "list": ["grass", "resource", "apple"]},
    "G": {"type": "all", "list": ["grass", "resource", "spawn_point"]},
    "D": "avatar_copy",
    "-": "inventory_display"
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
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "payHit"
            }
        },
        {
            "component": "BeamBlocker",
            "kwargs": {
                "beamType": "claimHit"
            }
        },
    ]
}

SAND = {
    "name": "sand",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "sand",
                "stateConfigs": [{
                    "state": "sand",
                    "layer": "background",
                    "sprite": "Sand",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["Sand"],
                "spriteShapes": [shapes.GRAINY_FLOOR],
                "palettes": [{"+": (222, 221, 189, 255),
                              "*": (219, 218, 186, 255)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
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

GRASS_EDGE = {
    "name": "grass_edge",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "grass_edge",
                "stateConfigs": [{
                    "state": "grass_edge",
                    "layer": "lowerPhysical",
                    "sprite": "GrassEdge",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["GrassEdge"],
                "spriteShapes": [shapes.GRASS_STRAIGHT_N_EDGE],
                "palettes": [{"*": (158, 194, 101, 255),
                              "@": (170, 207, 112, 255),
                              "x": (0, 0, 0, 0)}],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
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

COPY_SPAWN_POINT = {
   "name": "spawnPoint",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "spawnPoint",
                "stateConfigs": [{
                    "state": "spawnPoint",
                    "layer": "alternateLogic",
                    "groups": ["copySpawnPoints"]
                }],
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SHADOW_W = {
    "name": "shadow_w",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "shadow_w",
                "stateConfigs": [{
                    "state": "shadow_w",
                    "layer": "overlay",
                    "sprite": "ShadowW",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ShadowW"],
                "spriteShapes": [shapes.SHADOW_W],
                "palettes": [shapes.SHADOW_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SHADOW_E = {
    "name": "shadow_e",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "shadow_e",
                "stateConfigs": [{
                    "state": "shadow_e",
                    "layer": "overlay",
                    "sprite": "ShadowE",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ShadowE"],
                "spriteShapes": [shapes.SHADOW_E],
                "palettes": [shapes.SHADOW_PALETTE],
                "noRotates": [False]
            }
        },
        {
            "component": "Transform",
        },
    ]
}

SHADOW_N = {
    "name": "shadow_n",
    "components": [
        {
            "component": "StateManager",
            "kwargs": {
                "initialState": "shadow_n",
                "stateConfigs": [{
                    "state": "shadow_n",
                    "layer": "overlay",
                    "sprite": "ShadowN",
                }],
            }
        },
        {
            "component": "Appearance",
            "kwargs": {
                "renderMode": "ascii_shape",
                "spriteNames": ["ShadowN"],
                "spriteShapes": [shapes.SHADOW_N],
                "palettes": [shapes.SHADOW_PALETTE],
                "noRotates": [False]
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
NOOP        = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
FORWARD     = {"move": 1, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
STEP_RIGHT  = {"move": 2, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
BACKWARD    = {"move": 3, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
STEP_LEFT   = {"move": 4, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
TURN_LEFT   = {"move": 0, "turn": -1, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
TURN_RIGHT  = {"move": 0, "turn":  1, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
FIRE_ZAP    = {"move": 0, "turn":  0, "fireZap": 1, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 0}
FIRE_CLEAN  = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 1, "fireClaim": 0, "eat": 0, "pay": 0}
FIRE_CLAIM  = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 1, "eat": 0, "pay": 0}
EAT         = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 1, "pay": 0}
PAY         = {"move": 0, "turn":  0, "fireZap": 0, "fireClean": 0, "fireClaim": 0, "eat": 0, "pay": 1}
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
    EAT,
    PAY,
)

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

GREY_PLAYER_COLOR_PALETTES = []
COLOR_PLAYER_PALETTES = []
BRUSH_PALETTES = []
for shade_of_grey in colors.greys_avatar_palette:
  GREY_PLAYER_COLOR_PALETTES.append(shapes.get_palette(shade_of_grey))
  BRUSH_PALETTES.append(get_brush_palette(shade_of_grey))

for human_readable_color in colors.human_readable:
  COLOR_PLAYER_PALETTES.append(shapes.get_palette(human_readable_color))
  BRUSH_PALETTES.append(get_brush_palette(human_readable_color))

def create_resource(roles: list) -> PrefabConfig:
  """Configure the prefab to use for all resource objects."""
  # Setup unique states corresponding to each player who can claim the resource.
  claim_state_configs = []
  claim_sprite_names = []
  claim_sprite_rgb_colors = []
  for player_idx, role in enumerate(roles):
    lua_player_idx = player_idx + 1
    player_color = colors.human_readable[player_idx] if role == 'learner' else colors.greys_avatar_palette[player_idx]
    wet_sprite_name = "Color" + str(lua_player_idx) + "ResourceSprite"
    claim_state_configs.append({
        "state": "claimed_by_" + str(lua_player_idx),
        "layer": "resourceLayer",
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
                  "initialState": "unclaimedGrass",
                  "stateConfigs": [
                      {"state": "unclaimedGrass",
                       "layer": "resourceLayer",
                       "sprite": "UnclaimedResourceSprite"},
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
                  "delayTillSelfRepair": 15,
                  "selfRepairProbability": 0.1,
              }
          },
      ]
  }
  return prefab

def create_resource_texture() -> PrefabConfig:
  """Configure the background texture for a resource. It looks like grass."""
  prefab = {
      "name": "resource_texture",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "grass",
                  "stateConfigs": [
                    {
                        "state": "grass",
                        "layer": "background",
                        "sprite": "grass"
                    },
                    {
                        "state": "dessicated",
                        "layer": "background",
                        "sprite": "floor"
                    },
                  ],
              }
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["grass","floor"],
                  "spriteShapes": [shapes.GRASS_STRAIGHT, shapes.GRAINY_FLOOR],
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
          {
              "component": "Transform",
          },
      ]
  }
  return prefab


def get_water():
  """Get an animated water game object."""
  layer = "upperPhysical"
  water = {
      "name": "water_{}".format(layer),
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "water_1",
                  "stateConfigs": [
                      {"state": "water_1",
                       "layer": layer,
                       "sprite": "water_1",
                       "groups": ["water"]},
                      {"state": "water_2",
                       "layer": layer,
                       "sprite": "water_2",
                       "groups": ["water"]},
                      {"state": "water_3",
                       "layer": layer,
                       "sprite": "water_3",
                       "groups": ["water"]},
                      {"state": "water_4",
                       "layer": layer,
                       "sprite": "water_4",
                       "groups": ["water"]},
                  ]
              }
          },
          {"component": "Transform",},
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": ["water_1", "water_2", "water_3", "water_4"],
                  "spriteShapes": [shapes.WATER_1, shapes.WATER_2,
                                   shapes.WATER_3, shapes.WATER_4],
                  "palettes": [{
                      "@": (66, 173, 212, 255),
                      "*": (35, 133, 168, 255),
                      "o": (34, 129, 163, 255),
                      "~": (33, 125, 158, 255),}] * 4,
              }
          },
          {
              "component": "Animation",
              "kwargs": {
                  "states": ["water_1", "water_2", "water_3", "water_4"],
                  "gameFramesPerAnimationFrame": 2,
                  "loop": True,
                  "randomStartFrame": False,
                  "group": "water",
              }
          },
      ]
  }
  return water


def create_apple_prefab(regrowth_radius=-1.0,  # pylint: disable=dangerous-default-value
                        regrowth_probabilities=[0, 0.0, 0.0, 0.0]):
  """Creates the apple prefab with the provided settings."""
  growth_rate_states = [
      {
          "state": "apple",
          "layer": "appleLayer",
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
              "component": "Harvestable",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
              }
          },
          {
              "component": "DensityRegrow",
              "kwargs": {
                  "liveState": "apple",
                  "waitState": "appleWait",
                  "radius": regrowth_radius,
                  "regrowthProbabilities": regrowth_probabilities,
                  "maxAppleGrowthRate": 1,
                  "thresholdDepletion": THRESHOLD_APPLE_DEPLETION,
                  "thresholdRestoration": THRESHOLD_APPLE_RESTAURATION,
              }
          },
      ]
  }

  return apple_prefab


def create_prefabs(roles: Sequence[str],
                   regrowth_radius=-1.0,
                   # pylint: disable=dangerous-default-value
                   regrowth_probabilities=[0, 0.0, 0.0, 0.0]) -> PrefabConfig:
  """Returns a dictionary mapping names to template game objects."""
  num_players = len(roles)
  prefabs = {
      "wall": WALL,
      "sand": SAND,
      "grass": GRASS,
      "grass_edge": GRASS_EDGE,
      "spawn_point": SPAWN_POINT,
      "inside_spawn_point": INSIDE_SPAWN_POINT,
      "copy_spawn_point": COPY_SPAWN_POINT,
      "shadow_w": SHADOW_W,
      "shadow_e": SHADOW_E,
      "shadow_n": SHADOW_N,
      "river": get_water(),
      "resource_texture": create_resource_texture(),
      "resource": create_resource(roles=roles),
      "avatar_copy": create_avatar_copy(roles=roles),
      "inventory_display": create_inventory_display()
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
              "component": "DirtSpawner",
              "kwargs": {
                  "dirtSpawnProbability": 0.0,
                  "delayStartOfDirtSpawning": 50,
              },
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
                         role: str,
                         age: int,
                         spawn_group: str) -> Dict[str, Any]:
  """Create an avatar object that always sees itself as blue."""
  # Lua is 1-indexed.
  lua_index = player_idx + 1

  # Setup the self vs other sprite mapping.
  source_sprite_self = "Avatar" + str(lua_index)

  player_palette = COLOR_PLAYER_PALETTES[player_idx] if role == 'learner' else GREY_PLAYER_COLOR_PALETTES[player_idx]
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
                      {"state": live_state_name,
                       "layer": "upperPhysical",
                       "sprite": source_sprite_self,
                       "contact": "avatar",
                       "groups": ["players"]},

                      # Player wait state used when they have been zapped out
                      {"state": "playerWait",
                       "groups": ["playerWaits"]},
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
                  "palettes": [{**player_palette, **paintbrush_palette}],
                  "noRotates": [True]
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
                                  "eat",
                                  "pay"],
                  "actionSpec": {
                      "move": {"default": 0, "min": 0, "max": len(_COMPASS)},
                      "turn": {"default": 0, "min": -1, "max": 1},
                      "fireZap": {"default": 0, "min": 0, "max": 1},
                      "fireClean": {"default": 0, "min": 0, "max": 1},
                      "fireClaim": {"default": 0, "min": 0, "max": 1},
                      "eat": {"default": 0, "min": 0, "max": 1},
                      "pay": {"default": 0, "min": 0, "max": 1},
                  },
                  "view": {
                      "left": 5,
                      "right": 5,
                      "forward": 9,
                      "backward": 1,
                      "centered": False
                  },
              }
          },
          {
             "component": "Age",
              "kwargs": {
                  "age": age,
                  "max_life_span": DEFAULT_MAX_LIFE_SPAN,
              }
          },
          {
             "component": "Property",
              "kwargs": {
                  "playerIndex": lua_index,
                  "radius": 2,
              }
          },
          {
              "component": "Eating",
              "kwargs": {
                  "rewardForEating": 1.0,
              }
          },
          {
              "component": "Paying",
              "kwargs": {
                  "amount": 1.0,
                  "beamLength": 3,
                  "beamRadius": 2,
                  "agentRole": role,
              }
          },
          {
              "component": "Surroundings",
              "kwargs": {
                  "observationRadius": OBSERVATION_RADIUS,
                  "mapSize": MAP_SIZE,
                  "agentRole": role,
                  "agentLook": ROLE_SPRITE_DICT[role],
              }
          },
          {
              "component": "AllNonselfCumulants",
          },
          {
              "component": "Cleaner",
              "kwargs": {
                  "cooldownTime": 2,
                  "beamLength": 3,
                  "beamRadius": 1,
              }
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
                  "color": player_palette["*"],
                  "playerIndex": lua_index,
                  "beamLength": 1,
                  "beamRadius": 0,
                  "beamWait": 0,
              }
          },
          {
            "component": "Inventory",
              "kwargs": {
                  "mapSize": MAP_SIZE,
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
                  "penaltyForBeingZapped": PENALTY_FOR_BEING_ZAPPED,
                  "rewardForZapping": 0,
                  "removeHitPlayer": REMOVE_HIT_PLAYER,
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
                          "name": "AGENT_CLEANED",
                          "type": "Doubles",
                          "shape": [],
                          "component": "Cleaner",
                          "variable": "player_cleaned",
                      },
                      {
                          "name": "AGENT_LOOK",
                          "type": "String",
                          "shape": [],
                          "component": "Surroundings",
                          "variable": "agentLook",
                      },
                      {
                          "name": "SINCE_AGENT_LAST_CLEANED",
                          "type": "Int32s",
                          "shape": [],
                          "component": "Cleaner",
                          "variable": "sinceLastCleaned",
                      },
                      {
                          "name": "SINCE_AGENT_LAST_PAID",
                          "type": "Int32s",
                          "shape": [],
                          "component": "Paying",
                          "variable": "sinceLastPayed",
                      },
                      {
                          "name": "ALWAYS_PAYING_TO",
                          "type": "tensor.Int32Tensor",
                          "shape": [],
                          "component": "Paying",
                          "variable": "payingTo",
                      },
                      {
                          "name": "ALWAYS_PAID_BY",
                          "type": "Int32s",
                          "shape": [],
                          "component": "Paying",
                          "variable": "paidBy",
                      },
                      {
                          "name": "TIME_TO_GET_PAID",
                          "type": "Int32s",
                          "shape": [],
                          "component": "Paying",
                          "variable": "timeToGetPayed",
                      },
                      {
                          "name": "TOTAL_NUM_CLEANERS",
                          "type": "Int32s",
                          "shape": [],
                          "component": "Cleaner",
                          "variable": "num_cleaners",
                      },
                      {
                        "name": "INVENTORY",
                          "type": "Int32s",
                          "shape": [],
                          "component": "Inventory",
                          "variable": "inventory",
                      },
                      {
                          "name": "STOLEN_RECORDS",
                          "type": "Doubles",
                          "shape": [],
                          "component": "Property",
                          "variable": "got_robbed_by"
                      },
                      {
                          "name": "PROPERTY",
                          "type": "tensor.Int32Tensor",
                          "shape": [OBSERVATION_RADIUS],
                          "component": "Surroundings",
                          "variable": "property"
                      },
                      {
                          "name": "DEAD_APPLE_RATIO",
                          "type": "Doubles",
                          "shape": [],
                          "component": "Surroundings",
                          "variable": "deadAppleRatio"
                      },
                      {
                          "name": "SURROUNDINGS",
                          "type": "tensor.Int32Tensor",
                          "shape": [OBSERVATION_RADIUS],
                          "component": "Surroundings",
                          "variable": "surroundings"
                      },
                  ]
              }
          },
      ]
  }
  return avatar_object

def create_marking_overlay(player_idx: int) -> Mapping[str, Any]:
  """Create a marking overlay object."""
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
                  "spriteNames": ["sprite_for_level_1"],
                  "spriteShapes": [MARKING_SPRITE],
                  "palettes": [get_marking_palette(0.0)],
                  "noRotates": [True] * 3
              }
          },
      ]
  }
  return marking_object

def create_avatar_copy(roles: list) -> Mapping[str, Any]:
  copy_state_configs = []
  copy_sprite_names = []
  copy_palette_colors = []
  sprite_shapes = []
  num_players = len(roles)
  for player_idx, role in enumerate(roles):
    # Lua is 1-indexed.
    lua_idx = player_idx + 1
    source_sprite_self = "Avatar" + str(lua_idx)
    player_palette = COLOR_PLAYER_PALETTES[player_idx] if role == 'learner' else GREY_PLAYER_COLOR_PALETTES[player_idx]
    copy_state_configs.append({
        "state": "copy_of_" + str(lua_idx),
        "layer": "upperPhysical",
        "sprite": source_sprite_self,
        "groups": ["avatarCopies"]
    })
    copy_sprite_names.append(source_sprite_self)
    sprite_shapes.append(ROLE_SPRITE_DICT[role])
    copy_palette_colors.append(player_palette)

  avatar_copy_object = {
    "name": "avatar_copy",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "emptyTile",
                  "stateConfigs": [
                      {"state": "emptyTile",
                       "layer": "upperPhysical",
                       "sprite": "UnclaimedResourceSprite"},
                  ] + copy_state_configs,
              }
          },
          {
              "component": "Transform",
          },
          {
              "component": "Appearance",
              "kwargs": {
                  "renderMode": "ascii_shape",
                  "spriteNames": copy_sprite_names,
                  "spriteShapes": sprite_shapes,
                  "palettes": copy_palette_colors,
                  "noRotates": [True] * num_players
              }
          },
          {
              "component": "AvatarCopy",
              "kwargs": {}
          }
      ]
  }

  return avatar_copy_object

def create_inventory_display() -> Mapping[str, Any]:

    prefab = {
    "name": "inventory_display",
      "components": [
          {
              "component": "StateManager",
              "kwargs": {
                  "initialState": "appleWait",
                  "stateConfigs": [{
                        "state": "apple",
                        "layer": "upperPhysical",
                        "sprite": "Apple",
                    },
                    {
                        "state": "appleWait",
                        "layer": "logic",
                        "sprite": "AppleWait",
                    },]
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
              "component": "InventoryDisplay",
              "kwargs": {}
          }
      ]
  }
    return prefab

def create_avatar_and_associated_objects(roles: list, ages: list):
  """Returns list of avatars and their associated 
  marking objects of length 'num_players'."""
  avatar_objects = []
  additional_objects = []
  for player_idx, role in enumerate(roles):

    spawn_group = "spawnPoints"
    if player_idx < 2:
      # The first two player slots always spawn closer to the apples.
      spawn_group = "insideSpawnPoints"

    game_object = create_avatar_object(player_idx,
                                       role,
                                       ages[player_idx],
                                       spawn_group=spawn_group,
                                       )
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
      "READY_TO_SHOOT",

      # Cumulants.
      "AGENT_CLEANED",
      "AGENT_LOOK",
      "PROPERTY",
      "DEAD_APPLE_RATIO",
      "INVENTORY",
      "SURROUNDINGS",
      "TOTAL_NUM_CLEANERS",
      "SINCE_AGENT_LAST_CLEANED",
      "SINCE_AGENT_LAST_PAID",
      "ALWAYS_PAYING_TO",
      "ALWAYS_PAID_BY",
      "TIME_TO_GET_PAID",

      # Global observations
      "STOLEN_RECORDS",

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
      "SOURROUNDINGS": specs.surroundings(OBSERVATION_RADIUS),
      "POSITION": specs.OBSERVATION["POSITION"],
      "ORIENTATION": specs.OBSERVATION["ORIENTATION"],
      "WORLD.RGB": specs.rgb(168, 240),
  })

  # The roles assigned to each player.
  config.valid_roles = frozenset({"free",
                                  "cleaner", 
                                  "farmer",
                                  "learner",})
  # "bluie" as for one player
  config.default_player_roles = ("cleaner",) * 1 \
                                + ("farmer",) * 1 \
                                + ('free',) * 1 \
                                + ('learner',) * 1

  return config


def build(
    roles: Sequence[str],
    config: config_dict.ConfigDict,
) -> Mapping[str, Any]:
  """Build substrate definition given player roles."""
  del config
  num_players = len(roles)
  age_range = int(DEFAULT_MAX_LIFE_SPAN / len(roles))
  ages = [0] * num_players
  if DEFAULT_MAX_LIFE_SPAN < 200:
    ages = list(range(5, len(roles)*age_range, age_range)) + [0]
  # Build the rest of the substrate definition.
  substrate_definition = dict(
      levelName="rule_obeying_harvest",
      levelDirectory="meltingpot/lua/levels",
      #env_seed=env_seed,
      numPlayers=num_players,
      # Define upper bound of episode length since episodes end stochastically.
      maxEpisodeLengthFrames=500,
      spriteSize=8,
      topology="BOUNDED",  # Choose from ["BOUNDED", "TORUS"],
      simulation={
          "map": ASCII_MAP,
          "gameObjects": create_avatar_and_associated_objects(roles, ages),
          "scene": create_scene(),
          "prefabs": create_prefabs(roles,
                                    APPLE_RESPAWN_RADIUS,
                                    REGROWTH_PROBABILITIES),
          "charPrefabMap": CHAR_PREFAB_MAP,
      },
  )
  return substrate_definition
