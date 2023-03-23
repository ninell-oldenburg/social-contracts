import numpy as np

import dm_env

from typing import Tuple

from meltingpot.python.utils.policies.pysmt_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies import policy


foreign_property = Symbol('CUR_CELL_IS_FOREIGN_PROPERTY', BOOL)
cur_cell_has_apple = Symbol('CUR_CELL_HAS_APPLE', BOOL)
agent_has_stolen = Symbol('AGENT_HAS_STOLEN', BOOL)
num_cleaners = Symbol('NUM_CLEANERS', REAL)
dirt_fraction = Symbol('DIRT_FRACTION', REAL)

POTENTIAL_OBLIGATIONS = [
    ObligationRule(GT(dirt_fraction, Real(0.6)), 'CLEAN_ACTION'),
    ObligationRule(LT(num_cleaners, Real(1)), 'CLEAN_ACTION'),
    ObligationRule(GT(Symbol('SINCE_LAST_PAYED', INT),\
                      Symbol('PAY_RHYTHM', INT)), "PAY_ACTION"),
    ObligationRule(GT(Symbol('SINCE_LAST_CLEANED', INT),\
                      Symbol('CLEAN_RHYTHM', INT)), 'CLEAN_ACTION'),
]

POTENTIAL_PROHIBITIONS = [
    ProhibitionRule(And(cur_cell_has_apple, LT(Symbol
                   ('NUM_APPLES_AROUND', INT), Int(3))), 'MOVE_ACTION'),
    ProhibitionRule(Not(And(Symbol('IS_AT_WATER', BOOL), 
                    Symbol('FACING_NORTH', BOOL))), 'CLEAN_ACTION'),
    ProhibitionRule(And(Not(agent_has_stolen), And(foreign_property, 
                    cur_cell_has_apple)), 'MOVE_ACTION'),
]

class RuleLearningAgent(policy.Policy):
    def __init__(self,
                 env: dm_env.Environment,
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS) -> None:
        
        self.action_spec = env.action_spec()[0]
        self.discount_spec = env.discount_spec()
        self.reward_spec = env.reward_spec()
        self.observation_spec = env.observation_spec()
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.num_all_rules = len(self.potential_obligations) +\
                             len(self.potential_prohibitions)
        self.rule_probabilities = np.array([0.5] * self.num_all_rules)

    def step(self,
             timestep: dm_env.TimeStep) -> Tuple[int, dm_env.TimeStep]:
        """See base class."""
        # update beliefs over rules given observations
        for rule, idx in enumerate(self.potential_obligations + self.potential_prohibitions):
            self.rule_probabilities[idx] = self.update_belief(rule, idx, timestep.observation)
        # TODO

        # compute next action given a list of observations
        action, next_timestep = self._puppeteer.step(timestep, prev_states)
        return action, next_timestep
    
    def update_belief(self, rule, index, observation) -> float:
        """Returns updated probability for a given rule."""
        prior = self.rule_probabilities[index]
        likelihood = self.get_likelihood(rule, observation)
        marginal = self.num_all_rules
        posterior = (prior * likelihood) / marginal
        return posterior
    
    def get_likelihood(self, rule, observation) -> int:
        """Return likelihood that the rule is true given the observation."""
        likelihood = 0
        # TODO
        return likelihood
