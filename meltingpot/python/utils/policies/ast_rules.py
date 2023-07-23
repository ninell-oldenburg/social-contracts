import ast
import types
from meltingpot.python.configs.substrates.rule_obeying_harvest__complete import ROLE_SPRITE_DICT

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