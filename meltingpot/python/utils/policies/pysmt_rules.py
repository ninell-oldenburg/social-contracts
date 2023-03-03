from pysmt.shortcuts import *
from pysmt.smtlib.script import SmtLibScript

import numpy as np

class Rules():
    def __init__(self):
        # define reoccurring variable
        foreign_property = Symbol('forgein_property', BOOL)
        has_apple = Symbol('HAS_APPLE', BOOL)
        clean_action = Symbol('CLEAN_ACTION', BOOL)
        dirt_fraction = Symbol('DIRT_FRACTION', REAL)
        cleaner_role = Symbol('cleaner_role', BOOL)
        farmer_role = Symbol('farmer_role', BOOL)
        apples_paid = Symbol('apples_paid', INT)

        # define rules
        self.rules = [
            # don't if <2 apples around
            Not(And(has_apple, LT(Symbol('NUM_APPLES_AROUND', INT), Int(3)))),
            # don't fire the cleaning beam if you're not close to the water
            Not(And(clean_action, Not(Symbol('IS_AT_WATER', BOOL)))),
            # every time the water gets too polluted, go clean the water
            Implies(GT(dirt_fraction, Real(0.6)), clean_action),]
        """
            # don't if forgein property and has apples 
            Not(And(foreign_property, has_apples)),
            # do if forgein property but person has stolen before
            And(And(foreign_property, has_apples),
                And(Symbol('agent_has_stolen', BOOL))),
            # every X turns, go clean the water
            Implies(Equals(Symbol('since_last_cleaned', INT), Symbol('cleaning_rhythm', INT)),
                    clean_action),
            # clean the water if less than Y agents are cleaning
            Implies(LT(Symbol('num_cleaners', INT), Symbol("Y", INT)), 
                    clean_action),
            # if I'm in the cleaner role, go clean the water
            Implies(cleaner_role, clean_action),
            # Stop cleaning if I'm not paid by farmer
            Implies(And(cleaner_role, Not(Symbol('paid_by_farmer', BOOL))), 
                    Not(clean_action)),
            # Pay cleaner with apples
            Implies(farmer_role, GT(apples_paid, Int(0))),
            # stop paying cleaner if they don't clean
            Implies(Not(Symbol('cleaner_cleans', BOOL)), Equals(apples_paid, Int(0)))
            ]
            """

    def check(self, observation):
        """Returns True if a given rule holds, False if not"""
        for rule in self.rules:
            if not self.holds(rule, observation):
                return False
        return True
    
    def holds(self, rule, observation):
        """Returns if a rule holds given a certain observation."""
        variables = rule.get_free_variables()
        substitutions = {v: self.get_property(v, observation) for v in variables}
        problem = rule.substitute(substitutions)
        return is_sat(problem)

    def get_property(self, property, observation):
        """Get the requested properties from the observations
        and cast to pySMT compatible type."""
        value = observation[str(property)]
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.reshape(-1,) # unpack numpy array
            value = value[0]
        if isinstance(value, np.int32):
            value = int(value) # cast dtypes
        elif isinstance(value, np.float64):
            value = float(value)
        if isinstance(value, bool):
            value = Bool(value)
        elif isinstance(value, int):
            value = Int(value)
        elif isinstance (value, float):
            value = Real(value)

        return value
    
if __name__=='__main__':
    rules = Rules()
    results = set()
    for rule in rules.rules:
        results.update(rules.read_properties(rule))
    print(results)
