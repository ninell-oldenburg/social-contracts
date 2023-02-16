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

from typing import Generic, Tuple, TypeVar

import dm_env
import dmlab2d

from meltingpot.python.utils.puppeteers.rule_obeying_agent import RuleObeyingAgent

# https://github.com/deepmind/meltingpot/blob/main/examples/rllib/utils.py

class RuleObeyingPolicy(RuleObeyingPuppet):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, 
               agent: RuleObeyingAgent,
               env: dmlab2d.Environment) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingPuppet: The puppet coming with certain rules.
    """
    self._agent = agent
    self._env = env

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

  def available_actions(self) -> list:
    """Return the available actions at a given timestep."""
    actions = []
    true_states = self._agent.get_satisfied_rules()
    for action in len(self.ACTIONS):
      if self.satisfies(action, true_states):
          actions.append(action)

    """Sort actions by reward."""
    actions = sorted(actions, key=lambda action : action[1].reward) 
    return actions

  # TODO: this is only a proof of concept
  def compute_reward(self, current_states, new_states, action_name):
    # Compute the reward for the given action
    reward = 0
    if action_name == 'pick_apple':
      if current_states['cell.apple'] and not new_states['cell.apple']:
        reward += 1
    elif action_name == 'eat_apple':
      if current_states['agent.my_property'] and current_states['cell.apple']:
        reward += 1
    elif action_name == 'clean_water':
      if current_states['cell.water'] == 'polluted' and new_states['cell.water'] == 'clean':
        reward += 1

    return

    # TODO: TBD
    """
      def forward_bfs(self, timestep) -> int:
          # Perform a breadth-first search to generate the best plan
          best_plan = []
          queue = []
          while queue:
            node = queue.pop(0)
            if node['depth'] >= self.max_depth:
                best_plan = node['actions']
                break
            for action_name, action_value in self.ACTIONS.items():
                new_states = dict(self._agent.STATES)
                new_states_copy = dict(self._agent.STATES)
                new_states_copy['global.steps'] += 1
                new_states_copy['agent.clean_actions'] = new_states_copy['agent.clean_actions'][:-1] + [0]
                new_states[action_name] = action_value
                reward = self.compute_reward(new_states_copy, new_states, action_name)
                queue.append({'state': new_states, 
                'actions': node['actions'] + [action_name], 
                'depth': node['depth'] + 1, 
                'reward': node['reward'] + reward})


          plan = []
          state = self.initial_state()
          timestep = self._env.reset()

          # is the timestep calculated internally
          # i.e., do we not need to call it initially?
          queue = [(timestep, state, plan)]
          while queue:
            this_timestep, state, plan = queue.pop(0)
            
            if this_timestep.last():
              # we don't actually need to return a plan here do we?
              return plan

            # assumption: available actions only depend on the state
            avaiable_actions = self.available_actions(state)
            for action in avaiable_actions:
              # again, do we need to explicitly calculate the timestep here?
              next_timestep, next_state = self.step(this_timestep, state, action)
              next_plan =  [plan, action]
              queue.append((next_timestep, next_state, next_plan))

          return False
    """

  def step(self, 
           timestep: dm_env.TimeStep, 
           ) -> dm_env.TimeStep:
      """
      See base class.
      End of episode defined in dm_env.TimeStep.
      """
      # Check if episode has ended
      if timestep.last():
        return timestep

      # Get the list of rules that satisfy the current state
      satisfying_rules = self._agent.get_satisfied_rules()

      # If no rules are satisfied, do nothing
      if not satisfying_rules:
        return dm_env.TimeStep(
          step_type=dm_env.StepType.MID,
          reward=0,
          discount=1.0,
          observation=timestep.observation,
        )

      # Select an action based on the first satisfying rule
      action = self.select_action(satisfying_rules[0])

      return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=timestep.reward,
            discount=1.0,
            observation=timestep.observation,
            action=action,
        )
  
  def initial_state(self) -> Tuple[()]:
    """See base class."""
    return (self._agent.initial_state())

  def close(self) -> None:
    """See base class."""
    self._agent.close()
