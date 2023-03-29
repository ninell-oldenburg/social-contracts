import dm_env

from typing import Tuple
from collections import deque

from pysmt.shortcuts import *

import numpy as np

from meltingpot.python.utils.policies.pysmt_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy


foreign_property = Symbol('CUR_CELL_IS_FOREIGN_PROPERTY', BOOL)
cur_cell_has_apple = Symbol('CUR_CELL_HAS_APPLE', BOOL)
agent_has_stolen = Symbol('AGENT_HAS_STOLEN', BOOL)
num_cleaners = Symbol('NUM_CLEANERS', INT)
dirt_fraction = Symbol('DIRT_FRACTION', REAL)
cleaner_role = Symbol('CLEANER_ROLE', BOOL)
farmer_role = Symbol('FARMER_ROLE', BOOL)

POTENTIAL_OBLIGATIONS = [
    ObligationRule(GT(dirt_fraction, Real(0.6)), 'CLEAN_ACTION'),
    ObligationRule(LT(num_cleaners, Int(1)), 'CLEAN_ACTION'),
    ObligationRule(GT(Symbol('SINCE_LAST_PAYED', INT),\
                      Int(1)), "PAY_ACTION", farmer_role),
    ObligationRule(GT(Symbol('SINCE_LAST_CLEANED', INT),\
                          Int(1)), 'CLEAN_ACTION', cleaner_role),
]

POTENTIAL_PROHIBITIONS = [
    ProhibitionRule(And(cur_cell_has_apple, LT(Symbol
                   ('NUM_APPLES_AROUND', INT), Int(3))), 'MOVE_ACTION'),
    ProhibitionRule(Not(And(Symbol('IS_AT_WATER', BOOL), 
                    Symbol('FACING_NORTH', BOOL))), 'CLEAN_ACTION'),
    ProhibitionRule(And(Not(agent_has_stolen), And(foreign_property, 
                    cur_cell_has_apple)), 'MOVE_ACTION'),
]

POTENTIAL_PERMISSION = [
    # TODO
]

class RuleLearningAgent(RuleObeyingPolicy):
    def __init__(self,
                 env: dm_env.Environment,
                 player_idx: int, 
                 other_agents_roles: list,
                 role: str = "free",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS,
                 potential_permissions: list = POTENTIAL_PERMISSION) -> None:
        
        self._index = player_idx
        self.role = role
        self.other_agents_roles = other_agents_roles
        self._max_depth = 30
        self.action_spec = env.action_spec()[0]
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_permission = potential_permissions
        self.obligations = []
        self.prohibitions = []
        self.permissions = []
        self.current_obligation = None
        self.current_permission = None
        self.potential_rules = self.potential_obligations + self.potential_prohibitions
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([0.2]*self.num_rules)
        # poisson distribution can be modified
        self.poisson_distribution = np.random.poisson(lam=5, size=self.num_rules)
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

    def update_beliefs(self, observations, other_agent_actions):
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for i, rule in enumerate(self.potential_rules):
            # Compute the likelihood of the rule
            posterior = self.compute_posterior(i,
                                              rule,
                                               observations, 
                                               other_agent_actions)
            
            self.rule_beliefs[i] = posterior

    def compute_posterior(self, 
                          index,
                          rule, 
                          observations, 
                          other_agent_actions):
        """Returns a posterior for a rule given an observation 
        and other agents' actions."""
        posterior = 0.0
        likelihood = 1.0
        for action, role in zip(other_agent_actions, self.other_agents_roles):
        # check if the action violates the rule
            if isinstance(rule, ProhibitionRule):
                if not rule.holds(observations, action):
                    likelihood *= 0.1  # 0 if rule is violated
                    break
            elif isinstance(rule, ObligationRule):
                if not rule.holds(observations, role):
                    likelihood = 0.0
                    break

            likelihood *= 0.9 # only for prohibition

        prior = self.rule_beliefs[index]
        marginal = prior * likelihood + (1 - prior) * (1 - likelihood)
        posterior = (likelihood * prior) / marginal

        return posterior
    
    def sample_rules(self, threshold) -> None:
        """Updates the rules given a fixed number of highest priors."""
        self.obligations = []
        self.prohibitions = []
        for i, belief in enumerate(self.rule_beliefs):
            if belief > threshold:
                rule = self.potential_rules[i]
                if isinstance(rule, ObligationRule):
                    self.obligations.append(rule)
                elif isinstance(rule, ProhibitionRule):
                    self.prohibitions.append(rule)
        
    def step(self,
             timestep: dm_env.TimeStep,
             other_agent_actions) -> Tuple[int, dm_env.TimeStep]:
        """Use the learned rules to determine the actions of the agent."""
        
        self.update_beliefs(timestep.observation, other_agent_actions)
        threshold = 0.6
        self.sample_rules(threshold)

        print("="*50)
        print("CURRENT RULES")
        print(self.obligations)
        print(self.prohibitions)
        print("="*50)

        # use parent class to compute best step
        action = super().step(timestep)
        return action


