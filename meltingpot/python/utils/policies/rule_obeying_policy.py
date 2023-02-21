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
"""Bot policy implementations."""

from typing import Tuple

import dm_env
import dmlab2d

from meltingpot.python.utils.puppeteers.rule_obeying_agent_v2 import RuleObeyingAgent, RuleObeyingAgentState
from meltingpot.python.utils.policies import policy

# https://github.com/deepmind/meltingpot/blob/main/examples/rllib/utils.py

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, 
               agent: RuleObeyingAgent,
               # env: dmlab2d.Environment
               ) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingPuppet: The puppet coming with certain rules.
    """
    self._agent = agent
    self._prev_action = 0
    self._max_depth = 10
    # self._env = env
    """
    Args:
      env: dmlab2d.Lab2d -
      observation_names: List<string>
      seed: int - seed the environment.
      """
    
    # Define Actions
    self.ACTIONS = {
            'go_north': 0,
            'go_east': 1,
            'go_south': 2,
            'go_west': 3,
            'pick_apple': 4,
            'eat_apple': 5,
            'clean_water': 6,
            'stop_cleaning': 7,
            'pay_cleaner': 8
        }

  def step(self, 
           timestep: dm_env.TimeStep,
           prev_state: RuleObeyingAgentState
           ) -> Tuple[int, policy.State]:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """

      # Select an action based on the first satisfying rule
      action, state = self.forward_bfs(timestep, prev_state)
      self._prev_action = action

      return action, state

  def forward_bfs(self, timestep, prev_state) -> int:
    """Perform a breadth-first search to generate plan."""
    plan = [[0]]
    queue = [(prev_state, plan)]
    step_count = 0
    while queue:
      this_state, this_plan = queue.pop(0)
      # maybe define depth here
      if timestep.last() or step_count == self._max_depth:
        """Return top-most action."""
        return this_plan[1], this_state[1]

      observations = {
        key: value
        for key, value in timestep.observation.items()
        if 'WORLD' not in key
      }

      # Get the list of rules that satisfy the current state
      avaiable_actions = self.available_actions()
      for action in avaiable_actions:
        # again, do we need to explicitly calculate the timestep here?
        # TODO: create _agent that does all the stepping
        next_state = self._agent.step(
                              observations,
                              # previous action 
                              prev_state,
                              prev_action=action,
                              prev_reward=timestep.reward
                              )
        next_plan =  [plan, action]
        queue.append((next_state, next_plan))

      step_count += 1

    return False

  def available_actions(self) -> list:
    """Return the available actions at a given timestep."""
    actions = []
    # satisfied_rules = self.get_satisfied_rules()
    for _ in range(len(self.ACTIONS)):
      # TODO create mapping of action to true rules
      # if action in satisfied_rules:
        actions.append(1)

    return actions

    """Sort actions by reward.
    actions = sorted(actions, key=lambda action : action[1].reward) 
    return actions
    """

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
  
  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()

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

  def can_eat_others_apples(self, agent_x: RuleObeyingAgent):
    """Return True if another agent ate your apples."""
    agent_ate_yours = agent_x in self.STATES['agent.enemmies']
    return self.should_not_visit_forgein_apple_property() \
           and not self.should_not_visit_low_apple_density_cell() \
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
  def should_pay_cleaner(self, agent_x: RuleObeyingAgent):
    """Return True if you are in the farmer role and should pay another agent with apples."""
    return self.STATES['agent.role'] == 'farmer' \
           and agent_x.STATES['agent.role'] == 'cleaner' \
           and agent_x.has_cleaned_in_last_x_steps()


# Map states
    """
    self.STATES = {
            # 'agent.enemmies': self._agent.observation['enemies'], # list of enemies
            # 'agent.role': self._agent.observation['role'], # 'farmer' / 'cleaner' / 'free'
            # 'agent.paid_by_farmer': self._agent.observation['paid'], # bool
            # 'agent.clean_actions': self._agent.obersvation['clean_actions'], # list
            'cell.apple': self._env.observation['apple'], # bool
            'cell.water': self._env.observation['water'], # 'polluted' / 'clean'
            'global.steps': self._env.observation['steps'], # step count
            # 'agent.my_property': self._agent.observation['my_property'], # bool
            }
            """

    """
    # Map rules that are derived from the agent class
    self.RULES = {
            'should_not_visit_forgein_apple_property': self.should_not_visit_forgein_apple_property(),
            'should_not_visit_low_apple_density_cell': self.should_not_visit_low_apple_density_cell(),
            'can_pick_up_apple': self.can_pick_up_apple(),
            'can_eat_others_apples': self.can_eat_others_apples(),
            'should_clean_based_on_pollution': self.should_clean_based_on_pollution(),
            'should_clean_based_on_num_turns': self.should_clean_based_on_num_turns(),
            'is_active_cleaner': self.is_active_cleaner(),
            'should_clean_based_on_num_other_cleaners': self.should_clean_based_on_num_other_cleaners(),
            'should_stop_cleaning': self.should_stop_cleaning(),
            'has_cleaned_in_last_x_steps': self.has_cleaned_in_last_x_steps(),
            'should_pay_cleaner': self.should_pay_cleaner(),
    }
    """