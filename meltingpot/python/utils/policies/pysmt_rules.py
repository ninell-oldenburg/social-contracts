from pysmt.shortcuts import *
# from pysmt.solvers.solver import Solver

import numpy as np

class ProhibitionRule():
    def __init__(self, precondition):
        self.precondition = precondition
        self.s = Solver()

    def holds(self, observation):
        """Returns if a rule holds given a certain observation."""
        variables = self.precondition.get_free_variables()
        substitutions = {v: self.get_property(v, observation) for v in variables}            
        problem = self.precondition.substitute(substitutions)
        is_sat_val = self.s.is_sat(problem)
        return is_sat_val

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
    
class ObligationRule(ProhibitionRule):
    def __init__(self, precondition, goal):
        self.precondition = precondition
        self.goal = goal
        self.s = Solver()
        self.obligations = {'CLEAN_ACTION': list(range(9)), "PAY_ACTION": [11]}

    def get_valid_actions(self):
        return self.obligations[self.goal]
              
    def satisfied(self, action):
        return action == self.goal