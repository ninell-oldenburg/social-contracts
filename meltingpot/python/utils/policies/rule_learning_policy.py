import dm_env

from collections import deque

import numpy as np

from copy import deepcopy

import random

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_W_FARMER_HAT,
   'learner': shapes.CUTE_AVATAR_W_STUDENT_HAT,
   }

class RuleLearningPolicy(RuleObeyingPolicy):
    def __init__(self,
                 env: dm_env.Environment,
                 player_idx: int,
                 player_looks: list,
                 role: str = "learner",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS
                 ) -> None:
        
        self._index = player_idx
        self.role = role
        self._max_depth = 35
        self.action_spec = env.action_spec()[0]
        self.num_actions = self.action_spec.num_values
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_rules = self.potential_obligations + self.potential_prohibitions
        self.obligations = []
        self.prohibitions = []
        self.current_obligation = None
        self.player_looks = player_looks
        self.num_total_agents = len(player_looks)
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([(0.2)]*self.num_rules)
        self.nonself_active_obligations = np.array([dict() for _ in range(self.num_total_agents)])
        self.others_history = deque(maxlen=5)
        self.history = deque(maxlen=5)

        # move actions
        self.action_to_pos = [
            [[0,0],[0,-1],[0,1],[-1,0],[1,0]], # N
            [[0,0],[1,0],[-1,0],[0,-1],[0,1]], # E
            [[0,0],[0,1],[0,-1],[1,0],[-1,0]], # S
            [[0,0],[-1,0],[1,0],[0,1],[0,-1]], # W
            # N    # F    # SR  # B   # SL
          ]
        # turn actions
        self.action_to_orientation = [
            [3, 1], # N
            [0, 2], # E
            [1, 3], # S
            [2, 0], # W
          ]
        # non-move actions
        self.action_to_name = [
            "ZAP_ACTION",
            "CLEAN_ACTION",
            "CLAIM_ACTION",
            "EAT_ACTION",
            "PAY_ACTION"
          ]
        
        self.action_dict = {
            "NOOP_ACTION": [0],
            "MOVE_ACTION": [1, 2, 3, 4],
            "TURN_ACTION": [5, 6],
            "ZAP_ACTION": [7],
            "CLEAN_ACTION": [8],
            "CLAIM_ACTION": [9],
            "EAT_ACTION": [10],
            "PAY_ACTION": [11]
        }
        
    def update_beliefs(self, own_obs, other_agent_actions):
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for player_idx, action in enumerate(other_agent_actions):
        # check if the action violates the rule
            action_name = super().get_action_name(action)
            pos_list = self.get_position_list(own_obs)
            # available actions based on the current prohibitions
            past_available_actions = super().available_actions(self.others_history[-2][player_idx])
        
            # Compute the posterior of each rule
            self.compute_posterior(player_idx, action, action_name, past_available_actions, pos_list)

        # print(self.rule_beliefs)

    def compute_posterior(self, player_idx, action,
                          action_name, past_available_actions, pos_list) -> None:
        """Writes the posterior for a rule given an observation 
        and other agents' actions."""

        for rule_index, rule in enumerate(self.potential_rules):

            if isinstance(rule, ProhibitionRule):
                log_llh = self.comp_prohib_llh(action, action_name, rule, player_idx,
                                                past_available_actions)
    
            elif isinstance(rule, ObligationRule):
                    log_llh = self.comp_oblig_llh(player_idx, rule, pos_list,
                                                past_available_actions)
                        
            # do bayesian updating
            prior = self.rule_beliefs[rule_index]
            log_prior = np.log(prior)
            log_marginal = np.log(1/len(past_available_actions)) # num actions
            log_posterior = (log_prior + log_llh) - log_marginal
            posterior = np.exp(log_posterior)

            self.rule_beliefs[rule_index] = posterior
    
    def comp_oblig_llh(self, player_idx, rule, pos_list,
                       past_available_actions) -> np.log:
        player_role = self.get_role(player_idx)
        player_history = [all_players_timesteps[player_idx] for all_players_timesteps in self.others_history]

        pos = pos_list[player_idx]
        x, y = pos[0], pos[1]
        past_player_obs = super().update_observation(self.others_history[-2][player_idx], x, y)

        if rule.satisfied(past_player_obs, player_role):
            if rule in self.nonself_active_obligations[player_idx].keys():
                if self.nonself_active_obligations[player_idx][rule] <= self._max_depth:
                    # obligation satisfied
                    return np.log(0.9) # instead of 1
                else:
                    return np.log(1/len(past_available_actions)-1) # probably random action

        elif rule.holds_in_history(player_history, player_role):
            if rule in self.nonself_active_obligations[player_idx].keys():
                self.nonself_active_obligations[player_idx][rule] += 1
            else:
                self.nonself_active_obligations[player_idx][rule] = 0
        
        return np.log(1/(len(past_available_actions)))
    
    def comp_prohib_llh(self, action, action_name, rule, 
                        player_idx, past_available_actions) -> np.log:
    
        past_observation = self.others_history[-2][player_idx]
        past_pos = np.copy(past_observation['POSITION'])
        x, y, = super().update_coordinates_based_on_action(action, past_pos, past_observation)
        new_obs = super().update_observation(past_observation, x, y)

        if rule.holds(new_obs, action_name):
            return np.log(0.01) # violation
        else: # obedience
            if action_name == rule.prohibited_action: # precondition doesn't hold, rule can still be true
                return np.log(1/len(past_available_actions)-1)
            else: 
                return np.log(1/len(past_available_actions))
                
    def get_role(self, player_idx) -> str:
        for role in ROLE_SPRITE_DICT.keys():
            if self.player_looks[player_idx] == ROLE_SPRITE_DICT[role]:
                return role
        return "free"
    
    def threshold_rules(self, threshold):
        """Returns rules with probability over a certain threshold."""
        obligations = []
        prohibitions = []
        for i, belief in enumerate(self.rule_beliefs):
            if belief > threshold:
                rule = deepcopy(self.potential_rules[i])
                if isinstance(rule, ObligationRule):
                    obligations.append(rule)
                elif isinstance(rule, ProhibitionRule):
                    prohibitions.append(rule)

        return obligations, prohibitions
    
    def sample_rules(self) -> None:
        """Thompson samples rules."""
        obligations = []
        prohibitions = []
        prob = random.uniform(0, 1)
        for i, belief in enumerate(self.rule_beliefs):
            if belief > prob:
                rule = deepcopy(self.potential_rules[i])
                if isinstance(rule, ObligationRule):
                    obligations.append(rule)
                elif isinstance(rule, ProhibitionRule):
                    prohibitions.append(rule)

        return obligations, prohibitions
        
    def step(self,
             timestep: dm_env.TimeStep,
             other_agents_observations,
             other_agent_actions):
        """Use the learned rules to determine the actions of the agent."""

        observation = deepcopy(timestep.observation)
        self.others_history.append(other_agents_observations)
        self.history.append(observation)
        if len(self.others_history) >= 2:
            self.update_beliefs(observation, other_agent_actions)
        th_obligations, th_prohibitions = self.threshold_rules(threshold=0.5)
        sampl_obligations, sampl_prohibitions = self.sample_rules()

        # choose whether to use thresholded or sampled rules
        self.obligations = th_obligations
        self.prohibitions = th_prohibitions

        # """
        print('='*50)
        print('CURRENT RULES')
        print('Obligations:')
        for rule in self.obligations:
            print(rule.make_str_repr())
        print()
        print('Prohibitions:')
        for rule in self.prohibitions:
            print(rule.make_str_repr())
        print('='*50)
        # """

        # use parent class to compute best step
        return super().step(timestep)
    
    def get_position_list(self, observation) -> list:
        position_list = [None]*self.num_total_agents
        for i in range(observation['SURROUNDINGS'].shape[0]):
            for j in range(observation['SURROUNDINGS'].shape[1]):
                if observation['SURROUNDINGS'][i][j] > 0: # agent encountered
                    agent_idx = observation['SURROUNDINGS'][i][j] 
                    position_list[agent_idx-1] = (i, j)

        return position_list
