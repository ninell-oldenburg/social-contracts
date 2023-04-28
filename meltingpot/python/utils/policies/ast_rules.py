import ast
import types

class EnvironmentRule():
    def __init__(self, precondition):
        """Creates a new rule.

        Args:
            precondition: rule to be evaluated against
                the current timestep observation
        """
        self.precondition = "lambda obs : " + precondition

    def walk_lambda(self, ast_tree):
        # Wrap the lambda node in a complete expression
        expr = ast.Expression(body=ast_tree.body[0].value)
        # Compile the expression into a code object
        code = compile(expr, '<string>', 'eval')
        # Create the lambda function
        return types.FunctionType(code.co_consts[0], globals())

    def holds_precondition(self, obs):
        precondition_formula = self.walk_lambda(ast.parse(self.precondition))
        return precondition_formula(obs)
    

class ProhibitionRule(EnvironmentRule):
    """Contains rules that prohibit an action."""

    def __init__(self, precondition, prohibited_action):
        """See base class.

        Args:
            precondition: rule to be evaluated against
                the current timestep observation
            prohibited_action: action that is disallowed
        """

        self.precondition = "lambda obs : " + precondition
        self.prohibited_action = prohibited_action

    def holds(self, obs, action):
        """Returns True if a rule holds given a certain observation."""
        if not action == self.prohibited_action:
            return False
        
        return super().holds_precondition(obs)
    
    def make_str_repr(self):
        return self.precondition + ' -> !' + self.prohibited_action
    
class ObligationRule(EnvironmentRule):
    """Contains rules that emit a subgoal."""

    def __init__(self, precondition, goal, target_look):
        """See base class.

        Args:
            precondition: only checks for an environment state.
            goal: environment state to be achieved by the rule.
            role: role of the agent
        """

        self.precondition = "lambda obs : " + precondition
        self.goal = "lambda obs : " + goal
        self.target_look = target_look

    def holds_in_history(self, observations, look):
        """Returns True if a precondition holds given a certain vector of observation."""
        if self.target_look != look:
            return False
        
        for obs in observations:
            if super().holds_precondition(obs):
                return True
        
        return False
    
    def satisfied(self, observation, look):
        """Returns True if the rule goal is satisfied."""
        if self.target_look != look:
            return False
        
        goal_formula = super().walk_lambda(ast.parse(self.goal))
        return goal_formula(observation)
    
    def make_str_repr(self):
        return self.precondition + ' -> ' + self.goal