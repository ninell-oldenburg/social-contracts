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

import dm_env

class RuleObeyingAgent:
    # this can be made more elegantly
    def __init__(self, state: dm_env.TimeStep):
        self.state = state
        self._num_apples = state['global']['density_apples']
        self._num_cleaners = state['global']['num_cleaner_roles']
        # number of turns agents should clean to clean 'repeatedly'
        self._num_turns = state['global']['num_turns_repeated_clean']

        # Define the observations
        # TODO: fix observations (currently, this is definitely not how these work)
        self.STATES = {
            'cell.apple': self.state.observation['cell']['apple'], # bool
            'cell.water': self.state.observation['cell']['water'], # 'polluted' / 'clean'
            'global.steps': self.state.observation['global']['steps'], # step count
            'agent.my_property': self.state.observation['agent']['my_property'], # bool
            'agent.enemmies': self.state.observation['agent']['enemies'], # list of enemies
            'agent.role': self.state.observation['agent']['role'], # 'farmer' / 'cleaner' / 'free'
            'agent.paid_by_farmer': self.state.observation['agent']['paid'], # bool
            'agent.clean_actions': self.state.obersvation['agent']['clean_actions'] # list
            }

        self.RULES = [
            'should_not_visit_forgein_apple_property',
            'should_not_visit_low_apple_density_cell',
            'can_pick_up_apple',
            'can_eat_others_apples',
            'should_clean_based_on_pollution',
            'should_clean_based_on_num_turns',
            'is_active_cleaner',
            'should_clean_based_on_num_other_cleaners',
            'should_stop_cleaning',
            'has_cleaned_in_last_x_steps',
            'should_pay_cleaner',
        ]

        """
        Currently:
        observation_spec: Dict(
        NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP:Box(-inf, inf, (), float64), 
        NUM_OTHERS_WHO_ATE_THIS_STEP:Box(-inf, inf, (), float64), 
        NUM_OTHERS_WHO_CLEANED_THIS_STEP:Box(-inf, inf, (), float64), 
        ORIENTATION:Box(-2147483648, 2147483647, (), int32), 
        PLAYER_ATE_APPLE:Box(-inf, inf, (), float64), 
        PLAYER_CLEANED:Box(-inf, inf, (), float64), 
        POSITION:Box(-2147483648, 2147483647, (2,), int32), 
        READY_TO_SHOOT:Box(-inf, inf, (), float64), 
        RGB:Box(0, 255, (88, 88, 3), uint8))
        """

    # PROPERTY RULES
    def should_not_visit_forgein_apple_property(self):
        """Return True if agent shouldn't go to cells that have apples 
        and are not the agent's property."""
        return self.STATES['cell.apple'] and not self.STATES['agent.my_property']

    # HARVESTING RULES
    def should_not_visit_low_apple_density_cell(self):
        """Return True if the agent shouldn't go to cells with 
        low apple density."""
        return self.STATES['cell.apple'] and \
               self.STATES['cell.apple_count'] < self._num_apples

    def can_pick_up_apple(self):
        """Return True if agent can visit cell to pick up apple."""
        return not self.should_not_visit_forgein_apple_property() \
            and not self.should_not_visit_low_apple_density_cell()

    def can_eat_others_apples(self, agent_x):
        """Return True if another agent ate your apples."""
        agent_ate_yours = agent_x in self.STATES['agent.enemmies']
        return self.should_not_visit_forgein_apple_property() \
            and not self.should_not_visit_low_apple_density_cell(x) \
            and agent_ate_yours

    # CLEANING RULES
    def should_clean_based_on_pollution(self):
        """Return True if the agent should clean the water based on pollution."""
        return self.STATES['cell.water'] == 'polluted'

    def should_clean_based_on_num_turns(self):
        """Return True if the agent should clean the water every X turns."""
        return self.STATES['global.steps'] % self._num_turns == 0

    def is_active_cleaner(self):
        """Return True if the agent is in the cleaner role and gets paid."""
        return self.STATES['agent.paid_by_farmer'] and \
               self.STATES['agent.role'] == 'cleaner'

    def should_clean_based_on_num_other_cleaners(self):
        """Return True if the agent is obliged to clean the water 
        when maximum x other agents are cleaning it."""
        return self.STATES['cell.water'] == 'polluted' and \
               self.STATES['cell.cleaning_agents'] >= self._num_cleaners

    def should_stop_cleaning(self):
        """Return True if it is permitted to stop cleaning the water if the agent is not being paid by any "farmer" agents."""
        return self.STATES['agent.role'] == 'cleaner' and \
               not self.STATES['agent.paid_by_farmer']

    def has_cleaned_in_last_x_steps(self):
        """Return True if agent has cleaned in the last x turns."""
        return 1 in self.STATES['agent.clean_actions'][:-self._num_turns]

    # PAYING RULES
    def should_pay_cleaner(self, agent_x: RuleObeyingPuppet):
        """Return True if you are in the farmer role and should pay another agent with apples."""
        return self.STATES['agent.role'] == 'farmer' \
            and agent_x.STATES['agent.role'] == 'cleaner' \
            and agent_x.has_cleaned_in_last_x_steps()

    # GET SET OF RULES THAT ARE CURRENTLY SATISFIED
    def get_satisfied_rules(self):
        """Return a list of rules that are satisfied by the current state."""
        satisfied = []
        for rule in self.RULES:
            if self.satisfies(rule):
                satisfied.append(rule)
        return satisfied

    def satisfies(self, rule):
        """Return True if the given rule is satisfied by the current state."""
        if rule not in self.RULES:
            raise ValueError(f"Invalid rule: {rule}")

        return getattr(self, rule)()