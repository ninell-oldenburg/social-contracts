from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule

from meltingpot.python.utils.substrates import shapes

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_W_FARMER_HAT,
   'learner': shapes.CUTE_AVATAR_W_STUDENT_HAT,
   }

""" DEFAULT RULES """
""" OBLIGATIONS """
cleaning_precon_free = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 1"
cleaning_goal_free = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 1"
payment_precon_farmer = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 4"
payment_goal_farmer = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 1"
cleaning_precon_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 2"
cleaning_goal_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] < 1"
payment_precon_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 1"
payment_goal_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 0"

DEFAULT_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free, cleaning_goal_free, ROLE_SPRITE_DICT["free"]),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer, payment_goal_farmer, ROLE_SPRITE_DICT["farmer"]),
  # if you're a cleaner, wait until you've received a payment
  ObligationRule(payment_precon_cleaner, payment_goal_cleaner, ROLE_SPRITE_DICT["cleaner"]),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precon_cleaner, cleaning_goal_cleaner, ROLE_SPRITE_DICT["cleaner"]),
]

""" PROHIBITIONS """
harvest_apple_precon_standard = "lambda obs : obs['NUM_APPLES_AROUND'] < 2 and obs['CUR_CELL_HAS_APPLE']"
steal_from_forgein_cell_precon = "lambda obs : obs['CUR_CELL_HAS_APPLE'] and not obs['AGENT_HAS_STOLEN']"

DEFAULT_PROHIBITIONS = [
  # don't go if <2 apples around
  ProhibitionRule(harvest_apple_precon_standard, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precon, 'MOVE_ACTION'),
]

""" POTENTIAL RULES (TEST RULES) """
""" OBLIGATIONS """
cleaning_precon_free_2 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 2"
cleaning_goal_free_2 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 2"
cleaning_precon_free_3 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 3"
cleaning_goal_free_3 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 3"
cleaning_precon_free_4 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 4"
cleaning_goal_free_4 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 4"
cleaning_precon_free_5 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 5"
cleaning_goal_free_5 = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 5"

payment_precon_farmer_5 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 5"
payment_goal_farmer_5 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 5"
payment_precon_farmer_6 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 6"
payment_goal_farmer_6 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 6"
payment_precon_farmer_7 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 7"
payment_goal_farmer_7 = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] < 7"

cleaning_precon_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 4"
cleaning_goal_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] < 1"
cleaning_precon_cleaner_3 = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 3"
cleaning_goal_cleaner_2 = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] < 2"
cleaning_precon_cleaner_5 = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 5"
cleaning_goal_cleaner_3 = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] < 3"

payment_precon_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 1"
payment_goal_cleaner = "lambda obs : obs['TIME_TO_GET_PAYED'] == 0"
payment_precon_cleaner_0 = "lambda obs : obs['TIME_TO_GET_PAYED'] == 0"
payment_goal_cleaner_1 = "lambda obs : obs['TIME_TO_GET_PAYED'] == 1"

POTENTIAL_OBLIGATIONS = [
  ObligationRule(cleaning_precon_free, cleaning_goal_free, ROLE_SPRITE_DICT["free"]),
  ObligationRule(payment_precon_farmer, payment_goal_farmer, ROLE_SPRITE_DICT["farmer"]),
  ObligationRule(payment_precon_cleaner, payment_goal_cleaner, ROLE_SPRITE_DICT["cleaner"]),
  ObligationRule(cleaning_precon_cleaner, cleaning_goal_cleaner, ROLE_SPRITE_DICT["cleaner"]),
  # clean the water if less than 1 agent is cleaning
 #  ObligationRule(cleaning_precon_free, cleaning_goal_free, ROLE_SPRITE_DICT["free"]),
  ObligationRule(cleaning_precon_free_2, cleaning_goal_free_2, ROLE_SPRITE_DICT["free"]),
  ObligationRule(cleaning_precon_free_3, cleaning_goal_free_3, ROLE_SPRITE_DICT["free"]),
  ObligationRule(cleaning_precon_free_4, cleaning_goal_free_4, ROLE_SPRITE_DICT["free"]),
  ObligationRule(cleaning_precon_free_5, cleaning_goal_free_5, ROLE_SPRITE_DICT["free"]),
  # If you're in the farmer role, pay cleaner with apples
  # ObligationRule(payment_precon_farmer, payment_goal_farmer, ROLE_SPRITE_DICT["farmer"]),
  ObligationRule(payment_precon_farmer_5, payment_goal_farmer_5, ROLE_SPRITE_DICT["farmer"]),
  ObligationRule(payment_precon_farmer_6, payment_goal_farmer_6, ROLE_SPRITE_DICT["farmer"]),
  ObligationRule(payment_precon_farmer_7, payment_goal_farmer_7, ROLE_SPRITE_DICT["farmer"]),
  # If you're in the cleaner role, clean in a certain rhythm
  # ObligationRule(cleaning_precon_cleaner, cleaning_goal_cleaner, ROLE_SPRITE_DICT["cleaner"]),
  ObligationRule(cleaning_precon_cleaner_3, cleaning_goal_cleaner_2, ROLE_SPRITE_DICT["cleaner"]),
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_3, ROLE_SPRITE_DICT["cleaner"]),
  # if you're a cleaner, wait until you've received a payment
  # ObligationRule(payment_precon_cleaner, payment_goal_cleaner, ROLE_SPRITE_DICT["cleaner"]),
  ObligationRule(payment_precon_cleaner_0, payment_goal_cleaner_1, ROLE_SPRITE_DICT["cleaner"]),
]

""" PROHIBITIONS """
harvest_apple_precon_1 = "lambda obs : obs['NUM_APPLES_AROUND'] < 1 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_2 = "lambda obs : obs['NUM_APPLES_AROUND'] < 2 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_3 = "lambda obs : obs['NUM_APPLES_AROUND'] < 3 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_4 = "lambda obs : obs['NUM_APPLES_AROUND'] < 4 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_5 = "lambda obs : obs['NUM_APPLES_AROUND'] < 5 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_6 = "lambda obs : obs['NUM_APPLES_AROUND'] < 6 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_7 = "lambda obs : obs['NUM_APPLES_AROUND'] < 7 and obs['CUR_CELL_HAS_APPLE']"
harvest_apple_precon_8 = "lambda obs : obs['NUM_APPLES_AROUND'] < 8 and obs['CUR_CELL_HAS_APPLE']"
cur_cell_has_apple_precon = "lambda obs : obs['CUR_CELL_HAS_APPLE']"
position_equal_precon = "lambda obs: obs['POSITION'][0] == obs['POSITION'][1]"
total_num_cleaners_precon = "lambda obs: obs['TOTAL_NUM_CLEANERS'] == 1"
orientation_north_precon = "lambda obs: obs['ORIENTATION'] == 0"
orientation_east_precon = "lambda obs: obs['ORIENTATION'] == 1"
orientation_south_precon = "lambda obs: obs['ORIENTATION'] == 2"
orientation_west_precon = "lambda obs: obs['ORIENTATION'] == 3"

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
OBSERVATION SPACE =
['PROPERTY', 
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
'CUR_CELL_IS_FOREIGN_PROPERTY']
 """