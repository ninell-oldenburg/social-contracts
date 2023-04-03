import ast

class EnvironmentRule():
    def __init__(self, precondition):
        """Creates a new rule.

        Args:
            precondition: PySMT rule to be evaluated against
                the current timestep observation
        """
        self.precondition = precondition

    def walk_lambda(self, ast_tree):
        # Find the lambda function definition node
        lambda_node = None
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Lambda):
                lambda_node = node
                break

        # If a lambda node was found, evaluate it
        if lambda_node is not None:
            return exec(compile(ast.Expression(lambda_node), "<string>", "eval"))

    def holds(self, obs, rule_formula):
        rule_formula = self.precondition
        eval_expr = self.walk_lambda(rule_formula)
        if eval_expr is not None:
            return eval_expr(obs)
        return False
    
class ProhibitionRule(EnvironmentRule):
    """Contains rules that prohibit an action."""

    def __init__(self, precondition, prohibited_action):
        """See base class.

        Args:
            precondition: rule to be evaluated against
                the current timestep observation
            prohibited_action: action that is disallowed
        """

        self.precondition = precondition
        self.prohibited_action = prohibited_action

    def holds(self, obs, action):
        """Returns True if a rule holds given a certain observation."""
        if not action == self.prohibited_action:
            return False
        
        return super().holds(obs, rule_formula=self.precondition)
    
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

    def holds_in_history(self, observations, role):
        """Returns True if a precondition holds given a certain vector of observation."""
        if not self.role == role:
            return False
        
        for obs in observations:
            if not self.holds(obs, rule_formula=self.precondition):
                return False
        
        return True
    
    def satisfied(self, observation, role):
        """Returns True if the rule goal is satisfied."""
        if not self.role == role:
            return False
        
        return self.holds(observation, rule_formula=self.goal)
        

"""
foreign_property = Symbol('CUR_CELL_IS_FOREIGN_PROPERTY', BOOL)
cur_cell_has_apple = Symbol('CUR_CELL_HAS_APPLE', BOOL)
agent_has_stolen = Symbol('AGENT_HAS_STOLEN', BOOL)
num_cleaners = Symbol('TOTAL_NUM_CLEANERS', INT)
dirt_fraction = Symbol('DIRT_FRACTION', REAL)
sent_last_payment = Symbol('SINCE_AGENT_LAST_PAYED', INT)
did_last_cleaning = Symbol('SINCE_AGENT_LAST_CLEANED', INT)
received_last_payment = Symbol('SINCE_RECEIVED_LAST_PAYMENT', INT)

PYSMT_DEFAULT_OBLIGATIONS = [
    # clean the water if less than Y agents are cleaning
    ObligationRule(LT(num_cleaners, Int(1)), GE(num_cleaners, Int(1))),
    # If you're in the farmer role, pay cleaner with apples
    ObligationRule(GT(sent_last_payment, Int(1)), LE(sent_last_payment, Int(1)), 
                   "farmer"),
                      # If you're in the cleaner role, clean in a certain rhythm
    ObligationRule(GT(did_last_cleaning, Int(1)), LE(did_last_cleaning, Int(1)), 
                   "cleaner"),
    # if you're a cleaner, wait until you've received a payment
    ObligationRule(GT(received_last_payment, Int(1)), LE(received_last_payment, Int(1)), 
                    "cleaner"),
]

PYSMT_DEFAULT_PROHIBITIONS = [
    # don't go if <2 apples around
    ProhibitionRule(And(cur_cell_has_apple, LT(Symbol
                   ('NUM_APPLES_AROUND', INT), Int(3))), 'MOVE_ACTION'),
    # don't go if it is foreign property and cell has apples 
    ProhibitionRule(And(Not(agent_has_stolen), And(foreign_property, 
                    cur_cell_has_apple)), 'MOVE_ACTION'),
]
"""