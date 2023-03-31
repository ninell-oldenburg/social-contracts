import dm_env

from typing import Tuple
from collections import deque

from pysmt.shortcuts import *

import numpy as np

import random
from scipy.stats import beta
import math

from meltingpot.python.utils.policies.pysmt_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy


foreign_property = Symbol('CUR_CELL_IS_FOREIGN_PROPERTY', BOOL)
cur_cell_has_apple = Symbol('CUR_CELL_HAS_APPLE', BOOL)
agent_has_stolen = Symbol('AGENT_HAS_STOLEN', BOOL)
num_cleaners = Symbol('TOTAL_NUM_CLEANERS', INT)
dirt_fraction = Symbol('DIRT_FRACTION', REAL)
sent_last_payment = Symbol('SINCE_AGENT_LAST_PAYED', INT)
did_last_cleaning = Symbol('SINCE_AGENT_LAST_CLEANED', INT)
received_last_payment = Symbol('SINCE_RECEIVED_LAST_PAYMENT', INT)

POTENTIAL_OBLIGATIONS = [
    ObligationRule(LT(num_cleaners, Int(1)), GE(num_cleaners, Int(1))),
    ObligationRule(GT(sent_last_payment, Int(1)), LE(sent_last_payment, Int(1)), 
                   "farmer"),
    ObligationRule(GT(did_last_cleaning, Int(1)), LE(did_last_cleaning, Int(1)), 
                   "cleaner"),
    ObligationRule(GT(received_last_payment, Int(1)), LE(received_last_payment, Int(1)), 
                    "cleaner"),
]

POTENTIAL_PROHIBITIONS = [
    ProhibitionRule(And(cur_cell_has_apple, LT(Symbol
                   ('NUM_APPLES_AROUND', INT), Int(3))), 'MOVE_ACTION'),
    ProhibitionRule(And(Not(agent_has_stolen), And(foreign_property, 
                    cur_cell_has_apple)), 'MOVE_ACTION'),
]

POTENTIAL_PERMISSIONS = [
    # TODO
]

class RuleLearningPolicy(RuleObeyingPolicy):
    def __init__(self,
                 env: dm_env.Environment,
                 player_idx: int,
                 num_total_agents: int,
                 role: str = "free",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS
                 ) -> None:
        
        self._index = player_idx
        self.role = role
        self._max_depth = 30
        self.action_spec = env.action_spec()[0]
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_roles = ['free', 'cleaner', 'farmer', 'learner']
        self.obligations = []
        self.prohibitions = []
        self.nonself_active_obligations = []
        self.current_obligation = None
        self.num_total_agents = num_total_agents
        self.potential_rules = self.potential_obligations + self.potential_prohibitions
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([0.2]*self.num_rules)
        self.count_rule_tries = np.array([1]*self.num_rules)
        self.role_beliefs = np.array([[0.25]*4]*len(self.num_total_agents))
        # poisson distribution can be modified
        self.poisson_distribution = np.random.poisson(lam=5, size=self.num_rules)
        self.history = deque(maxlen=5)
        self.confidence_level = 1.0
        self.count_timestep = 0

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

    def update_beliefs(self, observations, other_agent_actions):
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for rule_index, rule in enumerate(self.potential_rules):
            # Compute the likelihood of the rule
            posterior = self.compute_posterior(rule_index,
                                               rule,
                                               observations, 
                                               other_agent_actions)
            
            self.rule_beliefs[rule_index] = posterior

    def compute_posterior(self, 
                          rule_index,
                          rule, 
                          observations, 
                          other_agent_actions):
        """Returns a posterior for a rule given an observation 
        and other agents' actions."""
        likelihood = 1.0
        for player_idx, action in enumerate(other_agent_actions):
        # check if the action violates the rule
            if isinstance(rule, ProhibitionRule):
                # TODO: observations currently don't specify the position of the agent
                # that is complying/not complying to the norm
                if not rule.holds(observations, action):
                    likelihood *= 0.1  # 0 if rule is violated
                else:
                    likelihood *= 0.9

            elif isinstance(rule, ObligationRule):
                if not player_idx == self._index:
                    for role_idx, role in enumerate(self.potential_roles):
                        if rule.holds(observations):
                            if not rule in self.nonself_active_obligations:
                                # if we encounter an obligation precondition, save it
                                self.nonself_active_obligations.append(rule)
                        if rule.satisfied(observations, role):
                            # if a previous precondition is fulfilled, likelihood increases
                            if rule in self.nonself_active_obligations:
                                likelihood *= 0.9
                                self.compute_role_likelihood(player_idx,
                                                             role_idx,
                                                             holds=True)
                            else:
                                likelihood *= 0.1
                                self.compute_role_likelihood(player_idx,
                                                             role_idx,
                                                             holds=False)

        prior = self.rule_beliefs[rule_index]
        marginal = prior * likelihood + (1 - prior) * (1 - likelihood)
        posterior = (prior * likelihood) / marginal

        return posterior
    
    def compute_role_likelihood(self, player_idx, role_idx, holds: bool) -> None:
        likelihood = 0.9 if holds == True else 0.1

        prior = self.role_beliefs[player_idx][role_idx]
        marginal = prior * likelihood + (1 - prior) * (1 - likelihood)
        posterior = (prior * likelihood) / marginal
            
        self.role_beliefs[player_idx][role_idx] = posterior

    # from https://gist.github.com/WhatIThinkAbout/235be75b217e8da40a4abe31d2f22c86#file-ucbsocket-py
    def uncertainty(self, t, i): 
        """ calculate the uncertainty in the estimate of this socket's mean """
        if self.count_rule_tries[i] == 0: return float('inf')                         
        return self.confidence_level * (np.sqrt(np.log(t) / self.count_rule_tries[i]))   
        
    
    def sample_rules(self, t, threshold) -> None:
        """Updates the rules given a fixed number of highest priors."""
        obligations = []
        prohibitions = []
        for i, belief in enumerate(self.rule_beliefs):
            ucb = belief + self.uncertainty(t, i)

            if ucb > threshold:
                rule = self.potential_rules[i]
                self.count_rule_tries[i] += 1
                if isinstance(rule, ObligationRule):
                    obligations.append(rule)
                elif isinstance(rule, ProhibitionRule):
                    prohibitions.append(rule)

        self.obligations = obligations
        self.prohibition = prohibitions
        
    def step(self,
             timestep: dm_env.TimeStep,
             other_agent_actions):
        """Use the learned rules to determine the actions of the agent."""
        self.count_timestep += 1

        self.update_beliefs(timestep.observation, other_agent_actions)
        self.sample_rules(self.count_timestep, threshold=0.5)

        print("="*50)
        print("CURRENT RULES")
        for x in self.obligations:
            print(x.precondition)
        for x in self.prohibitions:
            print(x.precondition)
        print("="*50)

        # use parent class to compute best step
        action = super().step(timestep)
        return action


