import dm_env

from collections import deque

import numpy as np

from copy import deepcopy

import random
from scipy.stats import bernoulli

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
                 other_player_looks: list,
                 log_output: bool,
                 look: shapes,
                 role: str = "learner",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS,
                 selection_mode: str = "threshold",
                 ) -> None:
        
        self._index = player_idx
        self.role = role
        self.look = look
        self.max_depth = 20
        self.p_obey = 0.9
        self.log_output = log_output
        self.selection_mode = selection_mode
        self.action_spec = env.action_spec()[0]
        self.num_actions = self.action_spec.num_values
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_rules = self.potential_obligations + self.potential_prohibitions
        self.obligations = []
        self.prohibitions = []
        self.current_obligation = None
        self.player_looks = other_player_looks
        self.num_focal_agents = self.get_num_focal(self.player_looks)
        self.num_total_agents = len(other_player_looks)
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([(0.2)]*self.num_rules)
        self.nonself_active_obligations_count = np.array([dict() for _ in range(self.num_total_agents)])
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
        
    def get_num_focal(self, player_looks):
        non_learners = 0
        for agent_look in player_looks:
            if agent_look != ROLE_SPRITE_DICT['learner']:
                non_learners += 1

        return non_learners
        
    def update_beliefs(self, other_actions):
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for player_idx in range(self.num_focal_agents):
            # Compute the posterior of each rule
            self.compute_posterior(player_idx, other_actions[player_idx])

        # print(self.rule_beliefs)

    def compute_posterior(self, player_idx, player_act) -> None:
        """Writes the posterior for a rule given an observation 
        and other agents' actions."""

        for rule_idx, rule in enumerate(self.potential_rules):

            # P(a | r = 1)
            if isinstance(rule, ProhibitionRule):
                log_llh = self.comp_prohib_llh(player_idx, rule, player_act)
    
            elif isinstance(rule, ObligationRule):
                log_llh = self.comp_oblig_llh(player_idx, rule, player_act)
                        
            # BAYESIAN UPDATING
            # P(r = 1)
            prior = self.rule_beliefs[rule_idx]
            log_prior = np.log(prior)
            # P(a) = P(a | r = 1) P(r = 1) + P(a | r = 0) P(r = 0)
            marginal = np.exp(log_llh) * prior + (1/self.num_actions) * (1 - prior)
            log_marginal = np.log(marginal)
            # P(r=1 | a) = P(a | r = 1) * P(r = 1) / P(a)
            log_posterior = (log_prior + log_llh) - log_marginal
            posterior = np.exp(log_posterior)

            self.rule_beliefs[rule_idx] = posterior
    
    def comp_oblig_llh(self, player_idx, rule, action) -> np.log:
        # unpack appearance, observation, position of the player
        player_look = self.player_looks[player_idx]
        player_history = [all_players_timesteps[player_idx] for all_players_timesteps in self.others_history]
        past_obs = self.others_history[-2][player_idx]

        past_pos = np.copy(past_obs['POSITION'])
        # transition to next possible observation
        x, y = super().update_coordinates_based_on_action(action, past_pos, past_obs)
        if self.exceeds_map(past_obs['WORLD.RGB'], x, y):
            return np.log(1/(self.num_actions)) # random action
    
        next_obs = super().update_observation(past_obs, x, y)

        if rule.holds_in_history(player_history, player_look):
            if rule not in self.nonself_active_obligations_count[player_idx].keys():
                self.nonself_active_obligations_count[player_idx][rule] = 0
            else:
                self.nonself_active_obligations_count[player_idx][rule] += 1
        # if rule is not active remove it from the dict
        elif rule in self.nonself_active_obligations_count[player_idx].keys():
             self.nonself_active_obligations_count[player_idx].pop(rule, None)

        if rule.satisfied(next_obs, player_look): # Agent obeyed the obligation
            if rule in self.nonself_active_obligations_count[player_idx].keys():
                if self.nonself_active_obligations_count[player_idx][rule] <= self.max_depth:
                    # P(a | rule = true) = P(a | obedient = true, rule = true) P(obedient = true) + ...
                    # P(obedient action | rule = true) = (1 * p_obey) + 1/n_actions * (1-p_obey)                     
                    # P(disobedient action | rule = true) = (0 * p_obey) + 1/n_actions * (1-p_obey)    
                    # Assume len(obligated_actions) = 1
                    p_obedient_action = self.p_obey + (1-self.p_obey)/(self.num_actions)
                    return np.log(p_obedient_action)
                else: # Rule has expired, i.e. agent disobeyed the obligation
                    p_disobedient_action = (1-self.p_obey)/(self.num_actions)
                    return np.log(p_disobedient_action)
            else: # obligation not active
                p_disobedient_action = (1-self.p_obey)/(self.num_actions)
                return np.log(p_disobedient_action)
        else: # action has not fulfilled the obligation
         return np.log(1/(self.num_actions))

                    
    def comp_prohib_llh(self, player_idx, rule, action) -> np.log:
    
        past_obs = self.others_history[-2][player_idx]
        past_pos = np.copy(past_obs['POSITION'])
        past_obs = self.update_observation(past_obs, past_pos[0], past_pos[1])

        prohib_actions = self.get_prohib_action(past_obs, rule, past_pos)
        num_prohib_acts = len(prohib_actions)
        if rule.holds_precondition(past_obs):
            if action in prohib_actions: # violation
                return np.log(0)
            else: # action not prohibited
                p_obedient_action = 1/(self.num_actions-num_prohib_acts)
                return np.log(p_obedient_action)
        else: # precondition doesn't hold
            p_obedient_action = 1/(self.num_actions-num_prohib_acts)
            return np.log(p_obedient_action)
    
    def get_prohib_action(self, observation, rule, cur_pos):
        prohib_acts = []
        for action in range(self.action_spec.num_values):
            x, y = self.update_coordinates_based_on_action(action, 
                                                            cur_pos,
                                                            observation)

            if self.exceeds_map(observation['WORLD.RGB'], x, y):
                prohib_acts.append(action)
                continue

            if observation['SURROUNDINGS'][x][y] == -2: # non-dirt water
                prohib_acts.append(action)
                continue

            new_obs = super().update_observation(observation, x, y)
            action_name = super().get_action_name(action)
            if rule.holds(new_obs, action_name):
                prohib_acts.append(action)
                continue

        return prohib_acts
    
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
    
    def append_history(self, timestep, other_obs):
        cur_obs = deepcopy(timestep.observation)
        self.others_history.append(other_obs)
        self.history.append(cur_obs)
        
    def step(self,
             timestep: dm_env.TimeStep,
             other_obs,
             other_acts):
        """Use the learned rules to determine the actions of the agent."""

        self.append_history(timestep, other_obs)
        if len(self.history) > 1:
            self.update_beliefs(other_acts)
        self.th_obligations, self.th_prohibitions = self.threshold_rules(threshold=0.95)
        self.sampl_obligations, self.sampl_prohibitions = self.sample_rules()

        # choose whether to use thresholded or sampled rules
        if self.selection_mode == 'threshold':
            self.obligations = self.th_obligations
            self.prohibitions = self.th_prohibitions
        else:
            self.obligations = self.sampl_obligations
            self.prohibitions = self.sampl_prohibitions

        # """
        if self.log_output:
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
