from pysmt.shortcuts import Symbol, And, Or, Not, get_model, LT, Int, GT

# TODO: make class useable

# Define the symbols for the state variables
Apple = Symbol("apple", bool)
ForgeinProperty = Symbol("forgein_property", bool)
AppleCount = Symbol("apple_count", int)
# Dirt = Symbol("dirt", int)
Steps = Symbol("steps", int)
# PaidByFarmer = Symbol("paid_by_farmer", bool)
# NumCleaners = Symbol("num_cleaners", int)

# Define the rules for the environment
# PROPERTY RULES
should_not_visit_forgein_apple_property = And(Apple, Not(ForgeinProperty))

# HARVESTING RULES
should_not_visit_low_apple_density_cell = And(Apple, LT(Int(3), AppleCount))
can_pick_up_apple = And(should_not_visit_forgein_apple_property, Not(should_not_visit_low_apple_density_cell))

"""
# CLEANING RULES
should_clean_based_on_pollution = GT(Dirt, 20)
should_clean_based_on_num_turns = (Steps % 20 == 0)
is_active_cleaner = PaidByFarmer
should_clean_based_on_num_other_cleaners = LT(NumCleaners, 2)
should_stop_cleaning = Not(PaidByFarmer)
"""

value_dict = {
"apple": True,
"forgein_property": True,
"apple_count": 3,
"steps": 20,
}

env.reset(value_dict)

"""
# FINAL EXPRESSION EVALUTION
is_sat = expr.simplify().substitute({"position": observations['POSITION'], "B": observations[1], "action": action}).is_true()
"""