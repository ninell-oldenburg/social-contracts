from pysmt.shortcuts import *
# from pysmt.solvers.solver import Solver

import numpy as np

class EnvironmentRule():
     """Parent class for one rule of the environment."""

     def __init__(self, precondition) -> None:
        """Creates a new rule.

        Args:
            precondition: PySMT rule to be evaluated against
                the current timestep observation
        """
        self.precondition = precondition
        self.s = Solver()

     def holds(self, observations):
        """Returns True if a rule holds given a certain vector of observation."""
        variables = self.precondition.get_free_variables()
        substitutions = {v: self.get_property(v, observations) for v in variables}            
        problem = self.precondition.substitute(substitutions)
        is_sat_val = self.s.is_sat(problem)
        return is_sat_val
     
     def get_property(self, property, observations):
        """Get the requested properties from the observations
        and cast to pySMT compatible type."""
        
        # INDEX THEM BY TIME
        # VARIABLE NAMES HAVE TO TAKE IN SOME SORT OF NAME
        # TIMESTEP 1.OBSERVATION

        value = observations[str(property)]
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

class ProhibitionRule(EnvironmentRule):
    """Contains rules that prohibit an action."""

    def __init__(self, precondition, prohibited_action):
        """See base class.

        Args:
            precondition: PySMT rule to be evaluated against
                the current timestep observation
            prohibited_action: action that is disallowed
        """

        self.precondition = precondition
        self.prohibited_action = prohibited_action
        self.s = Solver()

    def holds(self, observation, action):
        """Returns True if a rule holds given a certain observation."""
        if action == self.prohibited_action:
            variables = self.precondition.get_free_variables()
            substitutions = {v: self.get_property(v, observation) for v in variables}            
            problem = self.precondition.substitute(substitutions)
            return self.s.is_sat(problem)
        return False
    
class ObligationRule(EnvironmentRule):
    """Contains rules that emit a subgoal."""

    def __init__(self, precondition, goal):
        """See base class.

        Args:
            precondition: only checks for an environment state.
            goal: environment state to be achieved by the rule.
        """

        self.precondition = precondition
        self.goal = goal
        self.s = Solver()
              
    def satisfied(self, action):
        return action == self.goal
        """Returns True if a goal is achieved given a certain observation.
        variables = self.goal.get_free_variables()
        substitutions = {v: self.get_property(v, observation) for v in variables}            
        problem = self.goal.substitute(substitutions)
        is_sat_val = self.s.is_sat(problem)
        return is_sat_val"""