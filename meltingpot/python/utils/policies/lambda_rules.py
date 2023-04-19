from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule

""" DEFAULT RULES """
""" OBLIGATIONS """
cleaning_precondition_free = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 1"
cleaning_goal_free = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 1"
payment_precondition_farmer = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 4"
payment_goal_farmer = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 1"
cleaning_precondition_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 4"
cleaning_goal_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] < 1"
payment_precondition_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 1"
payment_goal_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 0"

DEFAULT_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precondition_free, cleaning_goal_free),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precondition_farmer, payment_goal_farmer, "farmer"),
  # if you're a cleaner, wait until you've received a payment
  ObligationRule(payment_precondition_cleaner, payment_goal_cleaner, "cleaner"),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precondition_cleaner, cleaning_goal_cleaner, "cleaner"),
]

""" PROHIBITIONS """
harvest_apple_precondition_standard = "lambda obs : obs['NUM_APPLES_AROUND'] < 2 and obs['CUR_CELL_HAS_APPLE']"
steal_from_forgein_cell_precondition = "lambda obs : obs['CUR_CELL_HAS_APPLE'] and not obs['AGENT_HAS_STOLEN']"

DEFAULT_PROHIBITIONS = [
  # don't go if <2 apples around
  ProhibitionRule(harvest_apple_precondition_standard, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precondition, 'MOVE_ACTION'),
]

""" POTENTIAL RULES (TEST RULES) """
""" OBLIGATIONS """
cleaning_precondition_free_2 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 2"
cleaning_goal_free_2 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 2"
cleaning_precondition_free_3 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 3"
cleaning_goal_free_3 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 3"
cleaning_precondition_free_4 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 4"
cleaning_goal_free_4 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 4"
cleaning_precondition_free_5 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 5"
cleaning_goal_free_5 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 5"

payment_precondition_farmer_5 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 5"
payment_goal_farmer_5 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 5"
payment_precondition_farmer_6 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 6"
payment_goal_farmer_6 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 6"
payment_precondition_farmer_7 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 7"
payment_goal_farmer_7 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 7"

cleaning_precondition_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 4"
cleaning_goal_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] < 1"
payment_precondition_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 1"
payment_goal_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 0"
# TBD

POTENTIAL_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precondition_free, cleaning_goal_free),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precondition_farmer, payment_goal_farmer, "farmer"),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precondition_cleaner, cleaning_goal_cleaner, "cleaner"),
  # if you're a cleaner, wait until you've received a payment
  ObligationRule(payment_precondition_cleaner, payment_goal_cleaner, "cleaner")
]

""" PROHIBITIONS """
harvest_apple_precondition_1 = "lambda obs : obs['NUM_APPLES_AROUND'] < 1 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_2 = "lambda obs : obs['NUM_APPLES_AROUND'] < 2 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_3 = "lambda obs : obs['NUM_APPLES_AROUND'] < 3 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_4 = "lambda obs : obs['NUM_APPLES_AROUND'] < 4 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_5 = "lambda obs : obs['NUM_APPLES_AROUND'] < 5 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_6 = "lambda obs : obs['NUM_APPLES_AROUND'] < 6 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_7 = "lambda obs : obs['NUM_APPLES_AROUND'] < 7 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precondition_8 = "lambda obs : obs['NUM_APPLES_AROUND'] < 8 and obs['CUR_CELL_HAS_APPLE']"
not_go_to_cell_precondition = "lambda obs : obs['CUR_CELL_HAS_APPLE']"


POTENTIAL_PROHIBITIONS = [
  # don't go if <x apples around
  ProhibitionRule(harvest_apple_precondition_1, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_2, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_3, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_4, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_5, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_6, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_7, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precondition_8, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precondition, 'MOVE_ACTION'),
]