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

import numpy as np

from meltingpot.python.utils.puppeteers.rule_obeying_agent_v2 import RuleObeyingAgent, RuleObeyingAgentState
from meltingpot.python.utils.policies import policy

import copy

# https://github.com/deepmind/meltingpot/blob/main/examples/rllib/utils.py

class RuleObeyingPolicy(policy.Policy):
  """A puppet policy controlled by a certain environment rules."""

  def __init__(self, 
               agent: RuleObeyingAgent,
               env,
               agent_id: int
               ) -> None:
    """Initializes the policy.

    Args:
      RuleObeyingAgent: Instantiate the RuleObeyingAgent class.
    """
    self._agent_id = agent_id
    self._agent = agent
    self._prev_action = 0
    self._max_depth = 10
    self._env = env
    self.ACTIONS = env.action_spec()[0]
    self.STATES = env.observation_spec
    self._action_simluation = [0] * 1 # substitute through elegantly getting the number of agents
    attrs = vars(self.STATES)
    #print(',\n'.join("%s: %s" % item for item in attrs.items()))
    #print(self.STATES)

    # currently this is relative to the agents' orientation
    """
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
    """
    
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
    queue = [(timestep, prev_state, plan)]
    step_count = 0
    while queue:
      this_timestep, this_state, this_plan = queue.pop(0)
      # maybe define depth here
      if this_timestep.last() or step_count == self._max_depth:
        """Return top-most action."""
        return this_plan[1], this_state

      # Get the current state of the environment
      observations = {
        key: value
        for key, value in this_timestep.observation[self._agent_id].items()
        if 'WORLD' not in key
      }

      # Get the list of rules that satisfy the current state
      avaiable_actions = self.available_actions(this_state)
      for action in avaiable_actions:
        # again, do we need to explicitly calculate the timestep here?
        # TODO: create _agent that does all the stepping
        self._action_simluation[self._agent_id] = action
        next_timestep = self._env.step(self._action_simluation)
        _, next_state = self._agent.step(
                            observations,
                            # previous action 
                            this_state,
                            prev_action=this_plan[0],
                            prev_reward=this_timestep.reward
                            )
        next_plan =  [plan, action]
        queue.append((next_timestep, next_state, next_plan))

      step_count += 1

    return False

  def available_actions(self, state) -> list:
    """Return the available actions at a given timestep."""
    actions = []
    for action in range(self.ACTIONS.num_values):
      self._action_simluation[self._agent_id] = action
      # TODO: check for actual logic via pySMT
      if self._env.step(self._action_simluation):
        actions.append(action)

    return actions

    """Sort actions by reward.
    actions = sorted(actions, key=lambda action : action[1].reward) 
    return actions
    """
  
  def initial_state(self) -> policy.State:
    """See base class."""
    return self._agent.initial_state()

  def close(self) -> None:
    """See base class."""
    self._agent.close()
