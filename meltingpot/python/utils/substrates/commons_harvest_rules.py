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
"""Game rules implementation in first-order logic."""
from typing import Generic, Tuple, TypeVar

import dm_env

# "O" for obligation, 
# "P" for permission, 
# and "F" for prohibition

RULES = {
    G(go_to_cell -> X (!apples_on_cell)),
    G((eat_apple(?a - agent, ?b - agent) | pick_up_apple(?a - agent, ?b - agent)) -> F (eat_apple(?b, ?a) | punish(?b, a))),
    G(polluted_water -> X (clean_water)),
    G((num_turn = X) -> X (clean_water)),
    G((num_cleaners < X) -> X (clean_water)),
    G((role = cleaner) -> F (clean_water)),
    G((paid_by_farmer = false) -> X (!clean_water)),
    G(pick_up_apple -> X ((num_apples_around >= X) | on_others_property = false)),
    G((is_others_property = true) -> X (!pick_up_apple)),
    G((role = cleaner) -> F (pay_with_apple)),
    G((cleans_repeatedly = false) -> X (!pay_with_apple))
},

def get_rule(state):
    for rules in RULES:
        return

# x is the coordinate of a cell
# a, b are variables for agents
STATES = {
    apples_on_cell(x) = ,
    polluted_water(x),
    role(a, u),
    paid_by(a, u),
    is_others_property(x),
    cleans_repeatedly(a)
},

ACTIONS = {
    go_to_cell(a, x),
    eat_apple(a, b),
    pick_up_apple(a, b),
    punish(a, b),
    clean_water(a, x),
    pay_with_apple(a, b)
}