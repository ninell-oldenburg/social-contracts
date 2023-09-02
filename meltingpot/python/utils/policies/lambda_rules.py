from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule

""" 
#################################################
################# DEFAULT RULES #################
################## OBLIGATIONS ################## 
#################################################
"""
cleaning_precon_free_30 = "obs['DIRT_FRACTION'] > 0.5 and obs['AGENT_LOOK'] == 0"
cleaning_goal_free_30 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"

payment_precon_farmer_15 = "obs['SINCE_AGENT_LAST_PAYED'] > 15 and obs['AGENT_LOOK'] == 2"
payment_goal_farmer_15 = "obs['SINCE_AGENT_LAST_PAYED'] == 0"

cleaning_precon_cleaner_5 = "obs['DIRT_FRACTION'] > 0.45 and obs['AGENT_LOOK'] == 1"
cleaning_goal_cleaner_5 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"

zap_precondition = "len(obs['RIOTS']) > 0"
zap_goal = "len(obs['RIOTS']) == 0"

DEFAULT_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5),
  # If you recognize that someone violated a rule, zap them
  # ObligationRule(zap_precondition, zap_goal)
]

CLEANING_RULES = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30).make_str_repr(),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15).make_str_repr(),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5).make_str_repr(),
]

""" 
################# PROHIBITIONS ################## 
"""
harvest_apple_precon_standard = "obs['CUR_CELL_HAS_APPLE'] and obs['NUM_APPLES_AROUND'] < 3"
steal_from_forgein_cell_precon = "obs['CUR_CELL_HAS_APPLE'] and obs['CUR_CELL_IS_FOREIGN_PROPERTY']"

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
cleaning_precon_free_5 = "obs['DIRT_FRACTION'] > 0.55 and obs['AGENT_LOOK'] == 0"
cleaning_goal_free_5 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"
cleaning_precon_free_10 = "obs['DIRT_FRACTION'] > 0.6 and obs['AGENT_LOOK'] == 0"
cleaning_goal_free_10 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"
cleaning_precon_free_15 = "obs['DIRT_FRACTION'] > 0.65 and obs['AGENT_LOOK'] == 0"
cleaning_goal_free_15 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"

payment_precon_farmer_5 = "obs['SINCE_AGENT_LAST_PAYED'] > 5 and obs['AGENT_LOOK'] == 2"
payment_goal_farmer_5 = "obs['SINCE_AGENT_LAST_PAYED'] == 0"
payment_precon_farmer_10 = "obs['SINCE_AGENT_LAST_PAYED'] > 10 and obs['AGENT_LOOK'] == 2"
payment_goal_farmer_10 = "obs['SINCE_AGENT_LAST_PAYED'] == 0"
payment_precon_farmer_30 = "obs['SINCE_AGENT_LAST_PAYED'] > 30 and obs['AGENT_LOOK'] == 2"
payment_goal_farmer_30 = "obs['SINCE_AGENT_LAST_PAYED'] == 0"

cleaning_precon_cleaner_10 = "obs['DIRT_FRACTION'] > 0.4 and obs['AGENT_LOOK'] == 1"
cleaning_goal_cleaner_10 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"
cleaning_precon_cleaner_15 = "obs['DIRT_FRACTION'] > 0.5 and obs['AGENT_LOOK'] == 1"
cleaning_goal_cleaner_15 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"
cleaning_precon_cleaner_30 = "obs['DIRT_FRACTION'] > 0.55 and obs['AGENT_LOOK'] == 1"
cleaning_goal_cleaner_30 = "obs['SINCE_AGENT_LAST_CLEANED'] == 0"

POTENTIAL_OBLIGATIONS = [
   # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precon_free_5, cleaning_goal_free_5),
  ObligationRule(cleaning_precon_free_10, cleaning_goal_free_10),
  ObligationRule(cleaning_precon_free_15, cleaning_goal_free_15),
  #ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precon_farmer_5, payment_goal_farmer_5),
  ObligationRule(payment_precon_farmer_10, payment_goal_farmer_10),
  #ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15),
  ObligationRule(payment_precon_farmer_30, payment_goal_farmer_30),
  # If you're in the cleaner role, clean in a certain rhythm
  #ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5),
  ObligationRule(cleaning_precon_cleaner_10, cleaning_goal_cleaner_10),
  ObligationRule(cleaning_precon_cleaner_15, cleaning_goal_cleaner_15),
  ObligationRule(cleaning_precon_cleaner_30, cleaning_goal_cleaner_30)
] + DEFAULT_OBLIGATIONS

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
  # ProhibitionRule(harvest_apple_precon_3, 'MOVE_ACTION'),
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
] + DEFAULT_PROHIBITIONS
