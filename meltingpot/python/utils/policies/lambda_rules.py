from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule

from meltingpot.python.configs.substrates.rule_obeying_harvest__complete import ROLE_SPRITE_DICT

""" 
########### APPREARANCE DEFINTIONS ############# 
"""

free = ROLE_SPRITE_DICT["free"]
farmer = ROLE_SPRITE_DICT["farmer"]
cleaner = ROLE_SPRITE_DICT["cleaner"]

""" 
#################################################
################# DEFAULT RULES #################
################## OBLIGATIONS ################## 
#################################################
"""
cleaning_precon_free_30 = "obs['SINCE_AGENT_LAST_CLEANED'] > 30"
cleaning_goal_free_30 = "obs['SINCE_AGENT_LAST_CLEANED'] < 30"
payment_precon_farmer_15 = "obs['SINCE_AGENT_LAST_PAYED'] > 15"
payment_goal_farmer_15 = "obs['SINCE_AGENT_LAST_PAYED'] < 15"
cleaning_precon_cleaner_5 = "obs['SINCE_AGENT_LAST_CLEANED'] > 5"
cleaning_goal_cleaner_5 = "obs['SINCE_AGENT_LAST_CLEANED'] < 5"

DEFAULT_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30, free),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15, farmer),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5, cleaner),
]

CLEANING_RULES = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30, free).make_str_repr(),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15, farmer).make_str_repr(),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5, cleaner).make_str_repr(),
]

""" 
################# PROHIBITIONS ################## 
"""
harvest_apple_precon_standard = "obs['NUM_APPLES_AROUND'] < 3 and obs['CUR_CELL_HAS_APPLE']"
steal_from_forgein_cell_precon = "obs['CUR_CELL_HAS_APPLE'] and not obs['AGENT_HAS_STOLEN']"

DEFAULT_PROHIBITIONS = [
  # don't go if <2 apples around
  ProhibitionRule(harvest_apple_precon_standard, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precon, 'MOVE_ACTION'),
]

PICK_APPLE_RULES = [
  # don't go if <2 apples around
  ProhibitionRule(harvest_apple_precon_standard, 'MOVE_ACTION').make_str_repr(),
]

TERRITORY_RULES = [
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precon, 'MOVE_ACTION').make_str_repr(),
]

""" 
#################################################
######## POTENTIAL RULES (TEST RULES) ###########
################## OBLIGATIONS ################## 
#################################################
"""
cleaning_precon_free_total = "obs['TOTAL_NUM_CLEANERS'] < 1"
cleaning_goal_free_total = "obs['TOTAL_NUM_CLEANERS'] > 1"
cleaning_precon_free_5 = "obs['SINCE_AGENT_LAST_PAYED'] > 5"
cleaning_goal_free_5 = "obs['SINCE_AGENT_LAST_PAYED'] < 5"
cleaning_precon_free_10 = "obs['SINCE_AGENT_LAST_PAYED'] > 10"
cleaning_goal_free_10 = "obs['SINCE_AGENT_LAST_PAYED'] < 10"
cleaning_precon_free_15 = "obs['SINCE_AGENT_LAST_PAYED'] > 15"
cleaning_goal_free_15 = "obs['SINCE_AGENT_LAST_PAYED'] < 15"

payment_precon_farmer_5 = "obs['SINCE_AGENT_LAST_PAYED'] > 5"
payment_goal_farmer_5 = "obs['SINCE_AGENT_LAST_PAYED'] < 5"
payment_precon_farmer_10 = "obs['SINCE_AGENT_LAST_PAYED'] > 10"
payment_goal_farmer_10 = "obs['SINCE_AGENT_LAST_PAYED'] < 10"
payment_precon_farmer_30 = "obs['SINCE_AGENT_LAST_PAYED'] > 30"
payment_goal_farmer_30 = "obs['SINCE_AGENT_LAST_PAYED'] < 30"

cleaning_precon_cleaner_10 = "obs['SINCE_AGENT_LAST_CLEANED'] > 10"
cleaning_goal_cleaner_10 = "obs['SINCE_AGENT_LAST_CLEANED'] < 10"
cleaning_precon_cleaner_15 = "obs['SINCE_AGENT_LAST_CLEANED'] > 15"
cleaning_goal_cleaner_15 = "obs['SINCE_AGENT_LAST_CLEANED'] <= 15"
cleaning_precon_cleaner_30 = "obs['SINCE_AGENT_LAST_CLEANED'] > 30"
cleaning_goal_cleaner_30 = "obs['SINCE_AGENT_LAST_CLEANED'] <= 30"

# Shoot if there is an apple in the current cell and the agent is ready to shoot
shooting_precon_apple = "obs['CUR_CELL_HAS_APPLE'] and obs['READY_TO_SHOOT']"
shooting_goal_apple = "not obs['CUR_CELL_HAS_APPLE']"

# If agent has not been paid for a long time, they must ask for payment
ask_payment_precon = "obs['SINCE_AGENT_LAST_PAYED'] > obs['TIME_TO_GET_PAYED']"
ask_payment_goal = "obs['ALWAYS_PAYED_BY']"

POTENTIAL_OBLIGATIONS = [
  # Shoot if there is an apple in the current cell and the agent is ready to shoot
  ObligationRule(shooting_precon_apple, shooting_goal_apple, free),
  # If agent has not been paid for a long time, they must ask for payment
  ObligationRule(ask_payment_precon, ask_payment_goal, cleaner),
   # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free_5, cleaning_goal_free_5, free),
  ObligationRule(cleaning_precon_free_10, cleaning_goal_free_10, free),
  ObligationRule(cleaning_precon_free_15, cleaning_goal_free_15, free),
  ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30, free),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer_5, payment_goal_farmer_5, free),
  ObligationRule(payment_precon_farmer_10, payment_goal_farmer_10, free),
  ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15, free),
  ObligationRule(payment_precon_farmer_30, payment_goal_farmer_30, free),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5, cleaner),
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5, free),
  ObligationRule(cleaning_precon_cleaner_10, cleaning_goal_cleaner_10, free),
  ObligationRule(cleaning_precon_cleaner_15, cleaning_goal_cleaner_15, free),
  ObligationRule(cleaning_precon_cleaner_30, cleaning_goal_cleaner_30, free)
]

""" 
################# PROHIBITIONS ################## 
"""
harvest_apple_precon_1 = "obs['NUM_APPLES_AROUND'] < 1 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_2 = "obs['NUM_APPLES_AROUND'] < 2 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_3 = "obs['NUM_APPLES_AROUND'] < 3 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_4 = "obs['NUM_APPLES_AROUND'] < 4 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_5 = "obs['NUM_APPLES_AROUND'] < 5 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_6 = "obs['NUM_APPLES_AROUND'] < 6 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_7 = "obs['NUM_APPLES_AROUND'] < 7 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_8 = "obs['NUM_APPLES_AROUND'] < 8 and obs['CUR_CELL_HAS_APPLE']"
cur_cell_has_apple_precon = "obs['CUR_CELL_HAS_APPLE']"
position_equal_precon = "obs['POSITION'][0] == obs['POSITION'][1]"
total_num_cleaners_precon = "obs['TOTAL_NUM_CLEANERS'] == 1"
orientation_north_precon = "obs['ORIENTATION'] == 0"
orientation_east_precon = "obs['ORIENTATION'] == 1"
orientation_south_precon = "obs['ORIENTATION'] == 2"
orientation_west_precon = "obs['ORIENTATION'] == 3"

POTENTIAL_PROHIBITIONS = [
  # don't go if <x apples around
  ProhibitionRule(harvest_apple_precon_1, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_2, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_3, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_4, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_5, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_6, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_7, 'MOVE_ACTION'),
  ProhibitionRule(harvest_apple_precon_8, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precon, 'MOVE_ACTION'),
  # don't eat. move, or turn if current cell has apple
  ProhibitionRule(cur_cell_has_apple_precon, 'EAT_ACTION'),
  ProhibitionRule(cur_cell_has_apple_precon, 'MOVE_ACTION'),
  ProhibitionRule(cur_cell_has_apple_precon, 'TURN_ACTION'),
  # if x == y, don't eat, move, or turn
  ProhibitionRule(position_equal_precon, 'EAT_ACTION'),
  ProhibitionRule(position_equal_precon, 'MOVE_ACTION'),
  ProhibitionRule(position_equal_precon, 'TURN_ACTION'),
  # if the total number of cleaner == 1, don't eat, move, or turn
  ProhibitionRule(total_num_cleaners_precon, 'EAT_ACTION'),
  ProhibitionRule(total_num_cleaners_precon, 'MOVE_ACTION'),
  ProhibitionRule(total_num_cleaners_precon, 'TURN_ACTION'),
  # dont move, turn, or eat if you're not looking north
  ProhibitionRule(orientation_north_precon, 'EAT_ACTION'),
  ProhibitionRule(orientation_north_precon, 'MOVE_ACTION'),
  ProhibitionRule(orientation_north_precon, 'TURN_ACTION'),
  # dont move, turn, or eat if you're not looking east
  ProhibitionRule(orientation_east_precon, 'EAT_ACTION'),
  ProhibitionRule(orientation_east_precon, 'MOVE_ACTION'),
  ProhibitionRule(orientation_east_precon, 'TURN_ACTION'),
  # dont move, turn, or eat if you're not looking south
  ProhibitionRule(orientation_south_precon, 'EAT_ACTION'),
  ProhibitionRule(orientation_south_precon, 'MOVE_ACTION'),
  ProhibitionRule(orientation_south_precon, 'TURN_ACTION'),
  # dont move, turn, or eat if you're not looking west
  ProhibitionRule(orientation_west_precon, 'EAT_ACTION'),
  ProhibitionRule(orientation_west_precon, 'MOVE_ACTION'),
  ProhibitionRule(orientation_west_precon, 'TURN_ACTION'),
]

"""
OBSERVATION SPACE = [
  'PROPERTY', 
  'SINCE_AGENT_LAST_CLEANED', 
  'POSITION', 
  'SINCE_AGENT_LAST_PAYED', 
  'AGENT_CLEANED', 
  'TOTAL_NUM_CLEANERS', 
  'READY_TO_SHOOT', 
  'SURROUNDINGS', 
  'ALWAYS_PAYING_TO', 
  'ALWAYS_PAYED_BY', 
  'TIME_TO_GET_PAYED', 
  'STOLEN_RECORDS', 
  'ORIENTATION', 
  'INVENTORY', 
  'WORLD.RGB',
  'NUM_APPLES_AROUND', 
  'CUR_CELL_HAS_APPLE', 
  'AGENT_HAS_STOLEN',
  'CUR_CELL_IS_FOREIGN_PROPERTY'
  ]
 """