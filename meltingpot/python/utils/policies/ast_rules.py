import ast
import types
import re
from meltingpot.python.utils.substrates import shapes

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_W_FARMER_HAT,
   'learner': shapes.CUTE_AVATAR,
   }

INT_TO_ROLE = {
    0: 'free',
    1: 'cleaner',
    2: 'farmer',
    3: 'learner'
}

ROLE_TO_INT = {
    'free': 0,
    'cleaner': 1,
    'farmer': 2,
    'learner': 3
}

class EnvironmentRule():
    def __init__(self, precondition):
        """Creates a new rule.

        Args:
            precondition: rule to be evaluated against
                the current timestep observation
        """
        self.precondition = "lambda obs : " + precondition
        self.precondition_formula = self.walk_lambda(ast.parse(self.precondition))

    def walk_lambda(self, ast_tree):
        # Wrap the lambda node in a complete expression
        expr = ast.Expression(body=ast_tree.body[0].value)
        # Compile the expression into a code object
        code = compile(expr, '<string>', 'eval')
        # Create the lambda function
        return types.FunctionType(code.co_consts[0], globals())

    def holds_precondition(self, obs):
        return self.precondition_formula(obs)
    
    def is_subset_of(self, other_rule):
        # Extract the conditions from the rules
        condition1 = re.findall(r"lambda obs\['(.*?)'\]", self.precondition)
        condition2 = re.findall(r"lambda obs\['(.*?)'\]", other_rule.precondition)

        print()
        print(condition1)
        print(condition2)
        
        # Check if the conditions are the same
        if condition1 != condition2:
            return False
        
       # Extract the numerical values and operators from the rules
        num_op1 = re.findall(r"([<>]) (\d+(\.\d+)?)", self.precondition)
        num_op2 = re.findall(r"([<>]) (\d+(\.\d+)?)", other_rule.precondition)

        # Check if either of the rules doesn't contain numerical values and operators
        if not num_op1 or not num_op2:
            return False

        op1, num1 = num_op1[0][0], float(num_op1[0][1])
        op2, num2 = num_op2[0][0], float(num_op2[0][1])

        # Check if the operators are the same
        if op1 != op2:
            return False
        
        # Check the numerical values based on the operator
        if op1 == '<':
            print()
            return float(num1) < float(num2)
        elif op1 == '>':
            return float(num1) > float(num2)

class ProhibitionRule(EnvironmentRule):
    """Contains rules that prohibit an action."""

    def __init__(self, precondition, prohibited_action):
        """See base class.

        Args:
            precondition: rule to be evaluated against
                the current timestep observation
            prohibited_action: action that is disallowed
        """

        self.pure_precon = precondition
        self.precondition = "lambda obs : " + precondition
        self.prohibited_action = prohibited_action
        self.precondition_formula = self.walk_lambda(ast.parse(self.precondition))

    def holds(self, obs, action):
        """Returns True if a rule holds given a certain observation."""
        if not action == self.prohibited_action:
            return False
        
        return super().holds_precondition(obs)
    
    def make_str_repr(self):
        return self.pure_precon + ' -> !' + self.prohibited_action
    
    def is_subset_of(self, other_rule):
        return super().is_subset_of(other_rule) and self.prohibited_action == other_rule.prohibited_action
    
class ObligationRule(EnvironmentRule):
    """Contains rules that emit a subgoal."""

    def __init__(self, precondition, goal):
        """See base class.
        
        Args:
            precondition: only checks for an environment state.
            goal: environment state to be achieved by the rule.
            role: role of the agent
        """

        self.pure_precon = precondition
        self.pure_goal = goal
        self.precondition = "lambda obs : " + precondition
        self.precondition_formula = self.walk_lambda(ast.parse(self.precondition))
        self.goal = "lambda obs : " + goal
        self.goal_formula = super().walk_lambda(ast.parse(self.goal))

    def holds_in_history(self, observations):
        """Returns True if a precondition holds given a certain vector of observation."""
        for i, obs in enumerate(observations):
            if super().holds_precondition(obs):
                for j in range(i, len(observations)):
                    if self.satisfied(observations[j]):
                        return False
                return True
        
        return False
    
    def satisfied(self, observation: dict) -> bool:
        """Returns True if the rule goal is satisfied."""
        return self.goal_formula(observation)
    
    def make_str_repr(self):
        return self.pure_precon + ' -> ' + self.pure_goal
    
    def is_subset_of(self, other_rule):
        return super().is_subset_of(other_rule) and self.goal == other_rule.goal