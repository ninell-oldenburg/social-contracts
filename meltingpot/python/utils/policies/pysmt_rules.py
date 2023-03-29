from pysmt.shortcuts import *
# from pysmt.solvers.solver import Solver

import numpy as np

from collections import deque

class EnvironmentRule():
     """Parent class for one rule of the environment."""
     solver = Solver()

     def __init__(self, precondition) -> None:
        """Creates a new rule.

        Args:
            precondition: PySMT rule to be evaluated against
                the current timestep observation
        """
        self.precondition = precondition

     def holds(self, observations):
        """Returns True if a rule holds given a certain vector of observation."""
        variables = self.precondition.get_free_variables()
        substitutions = {v: self.get_property(v, observations) for v in variables}            
        problem = self.precondition.substitute(substitutions)
        is_sat_val = EnvironmentRule.solver.is_sat(problem)
        return is_sat_val
     
     def get_property(self, property, observations):
        """Get the requested properties from the observations
        and cast to pySMT compatible type."""

        return self.cast(observations[property.symbol_name()])

     def cast(self, value):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.reshape(-1,) # unpack numpy array
            value = value[0]

        if isinstance(value, np.int32):
            value = int(value) 
        elif isinstance(value, np.float64):
            value = float(value)

        if isinstance(value, bool):
            value = Bool(value)
        elif isinstance(value, int):
            value = Int(value)
        elif isinstance(value, float):
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
        self.variables = self.precondition.get_free_variables()

    def holds(self, observation, action):
        """Returns True if a rule holds given a certain observation."""
        if action == self.prohibited_action:
            substitutions = {v: self.get_property(v, observation) for v in self.variables}            
            problem = self.precondition.substitute(substitutions)
            return EnvironmentRule.solver.is_sat(problem)
        return False
    
class ObligationRule(EnvironmentRule):
    """Contains rules that emit a subgoal."""

    def __init__(self, precondition, goal, role="free"):
        """See base class.

        Args:
            precondition: only checks for an environment state.
            goal: environment state to be achieved by the rule.
        """

        self.precondition = precondition
        self.goal = goal
        self.role = role
        self.variables = self.precondition.get_free_variables()

    def get_property(self, property, observations):
        """Returns the requested property from the observation history
        or a single observation dict.
        """
        if isinstance(observations, deque):
            return np.array([self.get_property(property, obs) for obs in observations])

        return super().get_property(property, observations)
    
    def holds_in_history(self, observations, role):
        """Returns True if a precondition holds given a certain vector of observation."""
        if self.role == role:
            for obs in observations:
                if not self.holds(obs):
                    return False
            return True
        return False
    
    def holds(self, observation):
        substitutions = {v: self.get_property(v, observation) for v in self.variables}
        problem = self.precondition.substitute(substitutions)
        return EnvironmentRule.solver.is_sat(problem)
              
    def satisfied(self, observation, role):
        """Returns True if the rule goal is satisfied."""
        if self.role == role:
            substitutions = {v: self.get_property(v, observation) for v in self.variables}
            problem = self.goal.substitute(substitutions)
            return EnvironmentRule.solver.is_sat(problem)
        return False

class PermissionRule(EnvironmentRule):
    """Contains rules that emit a subgoal."""

    def __init__(self, precondition, action_to_stop):
        """See base class.

        Args:
            precondition: only checks for an environment state.
            goal: environment state to be achieved by the rule.
        """

        self.precondition = precondition
        self.action_to_stop = action_to_stop
