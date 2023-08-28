import dm_env

from collections import deque

import numpy as np

from copy import deepcopy

import random

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.agent_timestep import AgentTimestep
from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS


class RuleLearningPolicy(RuleObeyingPolicy):
    def __init__(self,
                 env: dm_env.Environment,
                 player_idx: int,
                 # other_player_looks: list,
                 num_focal_bots: int,
                 log_output: bool,
                 look: shapes,
                 role: str = "learner",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS,
                 selection_mode: str = "threshold",
                 ) -> None:
        
        # CALLING PARAMETERS
        self._index = player_idx
        self.role = role
        self.look = look
        self.log_output = log_output
        self.action_spec = env.action_spec()[0]
        self.selection_mode = selection_mode
        self.num_actions = self.action_spec.num_values
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_rules = self.potential_prohibitions + self.potential_obligations

        # HYPERPARAMETER
        self.max_depth = 20
        self.compliance_cost = 0.1
        self.violation_cost = 0.3
        self.n_steps = 5
        self.gamma = 0.98
        self.n_rollouts = 10
        self.obligation_reward = 1
        self.initial_exp_r_cum = [30] * self.action_spec.num_values
        
        # GLOBAL INITILIZATIONS
        self.history = deque(maxlen=10)
        self.obligations = []
        self.prohibitions = []
        self.payees = []
        self.riots = []
        self.hash_table = {}
        if self.role == 'farmer':
            self.payees = None
        # TODO condition on set of active rules
        self.V = {'apple': {}, 'clean': {}, 'pay': {}, 'zap': {}} # nested policy dict
        self.ts_start = None
        self.goal = None
        self.x_max = 15
        self.y_max = 15

        # non-physical info
        self.last_zapped = 0
        self.last_payed = 0
        self.last_cleaned = 0

        # self.player_looks = other_player_looks
        self.num_focal_agents = num_focal_bots
        self.num_total_agents = num_focal_bots + 1 # TODO
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([(0.2)]*self.num_rules)
        self.nonself_active_obligations_count = np.array([dict() for _ in range(self.num_total_agents)])
        self.others_history = deque(maxlen=10)
        self.history = deque(maxlen=10)

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
        
    def update_beliefs(self, other_actions):
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for player_idx in range(self.num_focal_agents):
            # Compute the posterior of each rule
            past_ts = self.history[-2][player_idx]
            self.compute_posterior(player_idx, other_actions[player_idx], past_ts)

        # print(self.rule_beliefs)

    def compute_posterior(self, player_idx, player_act, past_ts) -> None:
        """Writes the posterior for a rule given an observation 
        and other agents' actions."""

        for rule_idx, rule in enumerate(self.potential_rules):

            # P(a | r = 1)
            if isinstance(rule, ProhibitionRule):
                log_llh = self.comp_prohib_llh(player_idx, rule, player_act)
    
            elif isinstance(rule, ObligationRule):
                log_llh = self.comp_oblig_llh(player_idx, rule, past_ts)
                        
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
    
    def comp_oblig_llh(self, player_idx: int, rule: ObligationRule) -> float:

        # unpack appearance, observation, position of the player
        # player_look = self.player_looks[player_idx]
        player_history = [all_players_timesteps.observation[player_idx] for all_players_timesteps in self.others_history]
        next_obs = self.others_history[-1].observation[player_idx]
        past_timestep = self.get_single_timestep(self.others_history[-2], player_idx)

        if rule.holds_in_history(player_history[:-2]):
            if rule not in self.nonself_active_obligations_count[player_idx].keys():
                self.nonself_active_obligations_count[player_idx][rule] = 0
            else:
                self.nonself_active_obligations_count[player_idx][rule] += 1
        # if rule is not active remove it from the dict
        elif rule in self.nonself_active_obligations_count[player_idx].keys():
             self.nonself_active_obligations_count[player_idx].pop(rule, None)

        # Check if obligation rule is active
        rule_active_count = self.nonself_active_obligations_count[player_idx].get(rule, float('inf'))
        rule_is_active = rule_active_count <= self.max_depth

        if rule_is_active: # Obligation is active
            if self.could_be_satisfied(rule, past_timestep, player_idx):
                try:
                    if rule.satisfied(next_obs): # Agent obeyed the obligation
                        # P(obedient action | rule = true) = (1 * p_obey) + 1/n_actions * (1-p_obey)
                        p_action = self.p_obey + (1-self.p_obey)/(self.num_actions)
                        return np.log(p_action)
                    else: # Agent disobeyed the obligation
                        # P(disobedient action | rule = true) = (0 * p_obey) + 1/n_actions * (1-p_obey)  
                        p_action = (1-self.p_obey)/(self.num_actions)
                        return np.log(p_action)
                except:
                    return np.log(1/(self.num_actions))
            else: # Obligation is not active, or has expired
                return np.log(1/(self.num_actions))
        else: # Obligation is not active, or has expired
            return np.log(1/(self.num_actions))

    def comp_prohib_llh(self, player_idx, rule, action) -> float:
    
        past_obs = self.others_history[-2].observation[player_idx]
        past_pos = np.copy(past_obs['POSITION'])
        past_obs = self.update_observation(past_obs, past_pos[0], past_pos[1])

        prohib_actions = self.get_prohib_action(past_obs, rule, past_pos)
        num_prohib_acts = len(prohib_actions)
        if rule.holds_precondition(past_obs):
            if action in prohib_actions: # violation
                p_action = (1-self.p_obey)/(self.num_actions)
                return np.log(p_action)
            else: # action not prohibited
                p_action = self.p_obey + (1-self.p_obey)/(self.num_actions-num_prohib_acts)
                # p_action = 1/(self.num_actions-num_prohib_acts)
                return np.log(p_action)
        else: # precondition doesn't hold
            p_action = 1/(self.num_actions-num_prohib_acts)
            return np.log(p_action)
        
    def get_prohib_action(self, observation, rule, cur_pos):
        prohib_acts = []
        for action in range(self.action_spec.num_values):
            x, y = self.update_coordinates_by_action(action, 
                                                     cur_pos,
                                                     observation)

            if self.exceeds_map(x, y):
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
    
    def get_single_timestep(self, env_timestep, idx):
        """Returns single agent timestep from environment timestep."""
        agent_timestep = dm_env.TimeStep(
            step_type=env_timestep.step_type,
            reward=env_timestep.reward[idx],
            discount=env_timestep.discount,
            observation=env_timestep.observation[idx])
        
        return agent_timestep
    
    def could_be_satisfied(self, rule, past_timestep, idx):
        """Returns True is an obligation could be satisfied."""
        for action in range(self.action_spec.num_values):
            next_timestep = super().env_step(past_timestep, action, idx)
            try:
                if rule.satisfied(next_timestep.observation):
                    return True
            except TypeError:
                return False
        return False
    
    def threshold_rules(self):
        """Returns rules with probability over a certain threshold."""
        obligations = []
        prohibitions = []
        for i, belief in enumerate(self.rule_beliefs):
            if belief >= self.threshold:
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
    
    def update_and_append_others_history(self, 
                       all_timestep: list[dm_env.TimeStep]):
        """Appends timestep observation to current 
        environent timestep to overall history."""

        for i, observation in enumerate(all_timestep.observation):
            cur_obs = super().deepcopy_dict(observation)
            cur_pos = np.copy(cur_obs['POSITION'])
            x, y = cur_pos[0], cur_pos[1]
            all_timestep.observation[i] = super().update_observation(cur_obs, x, y)

        self.others_history.append(all_timestep)
        
    def step(self,
             timestep: dm_env.TimeStep):
        """Use the learned rules to determine the actions of the agent."""

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
