from pysmt.shortcuts import *
# from pysmt.solvers.solver import Solver

import numpy as np
    
class ObligationRule():
    def __init__(self, precondition, goal):
        # precondiiton: what holds s.t. obligation needs to come into place
        self.precondition = precondition
        # goal: what should be done now
        self.goal = goal
        
    def holds(self, observation, action):
        # Check if precondition is satisfied / true
        # e.g. if dirtFraction is above some value
        pass
              
    def satisfied(self, observation, action):
        # Check if goal formula is satisfied
        # e.g. make dirtFraction go below some value
        pass

class ProhibitionRule():
    def __init__(self, formula):
        self.formula = formula
        self.s = Solver()

    def holds(self, observation):
        """Returns if a rule holds given a certain observation."""
        variables = self.formula.get_free_variables()
        substitutions = {v: self.get_property(v, observation) for v in variables}
        problem = self.formula.substitute(substitutions)
        return self.s.is_sat(problem)

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