from pysmt.shortcuts import Symbol, is_sat, And, Equals, Int
from pysmt.smtlib.parser import SmtLibParser
# from six.moves import cStringIO
from pysmt.typing import INT

import dm_env
import numpy as np

class Rule():
    def __init__(self):
        # current problem: I want to incrementally build the rule_properties
        # is there a function that iterates over symbols and saves them 
        PLAYER_ATE_APPLE = Symbol("PLAYER_ATE_APPLE", INT)
        self.rule = (Equals(PLAYER_ATE_APPLE, Int(3)))
        self.action_orientation = {
            "N": [[0,0],[0,-1],[0,1],[-1,0],[1,0]],
            "S": [[0,0],[0,1],[0,-1],[1,0],[-1,0]],
            "E": [[0,0],[1,0],[-1,0],[0,1],[0,-1]],
            "W": [[0,0],[-1,0],[1,0],[0,-1],[0,1]]
        }
        # define which properties are requested by the rule
        self.rule_properties = {"PLAYER_ATE_APPLE": 3}

        """    
        def parse_input(self):
            parser = SmtLibParser()
            script = parser.get_script(cStringIO(self.rule))
            for cmd in script:
                print(cmd)
        """
    
    def validate(self, observation, action):
        """Returns True if a given rule holds, False if not"""
        # for every key in the observation, check if there's some relating property in the observations
        for key in self.rule_properties.keys():
            if not key in observation[0].keys():
                raise AttributeError("Observation doesn't hold "
                                     "requested property")
        
        # then get the property values

        problem = And(observation, self.rule)
        # check if the property values hold the rule
        return is_sat(problem)

    def get_new_position(self, observation, action):
        """Returns a position based on action, orientation, and position"""
        position = observation['POSITION']
        orientation = observation['ORIENTATION']
        orientation_based_action = self.action_orientation[orientation][action]
        new_position = np.add(position, orientation_based_action)
    
def test_rule_checker():
    # Initialize a RuleChecker instance
    # Define a rule to check
    # input_rule = (Equals("PLAYER_ATE_APPLE", Int(3)))
    rule = Rule()

    timestep = dm_env.TimeStep(step_type=dm_env.StepType.MID,
                        reward=[0],
                        discount=1.0,
                        observation=[{'RGB': np.zeros((10, 10, 3), dtype=np.uint8),
                                      'POSITION': np.array([0, 0], dtype=np.int32),
                                      'ORIENTATION': np.array(0, dtype=np.int32),
                                      'NUM_OTHERS_WHO_CLEANED_THIS_STEP': np.array(0.0),
                                      'PLAYER_ATE_APPLE': np.array(0.0),
                                      'PLAYER_CLEANED': np.array(0.0),
                                      'NUM_OTHERS_WHO_ATE_THIS_STEP': np.array(0.0),
                                      'READY_TO_SHOOT': np.array(0.0),
                                      'NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP': np.array(0.0)}])

    action = 1 # move forward
    agent_id = 0
    agent_observation = timestep.observation[agent_id]
    assert rule.validate(agent_observation, action) == False

test_rule_checker()

"""
# Define the symbols for the state variables
Apple = Symbol("apple", bool)
ForgeinProperty = Symbol("forgein_property", bool)
AppleCount = Symbol("apple_count", int)
# Dirt = Symbol("dirt", int)
Steps = Symbol("steps", int)
# PaidByFarmer = Symbol("paid_by_farmer", bool)
# NumCleaners = Symbol("num_cleaners", int)

# Define the rules for the environment
# PROPERTY RULES
should_not_visit_forgein_apple_property = And(Apple, Not(ForgeinProperty))

# HARVESTING RULES
should_not_visit_low_apple_density_cell = And(Apple, LT(Int(3), AppleCount))
can_pick_up_apple = And(should_not_visit_forgein_apple_property, Not(should_not_visit_low_apple_density_cell))

# CLEANING RULES
should_clean_based_on_pollution = GT(Dirt, 20)
should_clean_based_on_num_turns = (Steps % 20 == 0)
is_active_cleaner = PaidByFarmer
should_clean_based_on_num_other_cleaners = LT(NumCleaners, 2)
should_stop_cleaning = Not(PaidByFarmer)

value_dict = {
"apple": True,
"forgein_property": True,
"apple_count": 3,
"steps": 20,
}

env.reset(value_dict)

# FINAL EXPRESSION EVALUTION
is_sat = expr.simplify().substitute({"position": observations['POSITION'], "B": observations[1], "action": action}).is_true()
"""