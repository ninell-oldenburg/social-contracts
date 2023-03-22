from meltingpot.python.utils.policies import policy
import dm_env
from typing import Tuple
from meltingpot.python.utils.policies.pysmt_rules import ProhibitionRule, ObligationRule

foreign_property = Symbol('CUR_CELL_IS_FOREIGN_PROPERTY', BOOL)
cur_cell_has_apple = Symbol('CUR_CELL_HAS_APPLE', BOOL)
agent_has_stolen = Symbol('AGENT_HAS_STOLEN', BOOL)
num_cleaners = Symbol('NUM_CLEANERS', REAL)
dirt_fraction = Symbol('DIRT_FRACTION', REAL)

POSSIBLE_RULES = [
    ObligationRule(GT(dirt_fraction, Real(0.6)), 'CLEAN_ACTION'),
    ObligationRule(LT(num_cleaners, Real(1)), 'CLEAN_ACTION'),
    ObligationRule(GT(Symbol('SINCE_LAST_PAYED', INT),\
                      Symbol('PAY_RHYTHM', INT)), "PAY_ACTION"),
    ObligationRule(GT(Symbol('SINCE_LAST_CLEANED', INT),\
                      Symbol('CLEAN_RHYTHM', INT)), 'CLEAN_ACTION'),
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
                 potential_norms: list = POTENTIAL_NORMS) -> None:
        
        self.action_spec = env.action_spec()[0]
        self.discount_spec = env.discount_spec()
        self.reward_spec = env.reward_spec()
        self.observation_spec = env.observation_spec()
        self.potential_norms = potential_norms

    def step(self,
             timestep: dm_env.TimeStep,
             prev_states: list) -> Tuple[int, dm_env.TimeStep]:
        """See base class."""
        # update beliefs over rules given observations


        # compute next action given a list of observations
        action, next_timestep = self._puppeteer.step(timestep, prev_states)
        return action, next_timestep
    
