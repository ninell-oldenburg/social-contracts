
from examples.eval_scripts.view_custom_model import main, ROLE_SPRITE_DICT
import pandas as pd
import time
import datetime

from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

import itertools

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS

# Generate all possible combinations of the rules
RULE_COMBINATIONS = [[]] # include empty rule set
for i in range(1, len(DEFAULT_RULES) + 1):
    RULE_COMBINATIONS += list(itertools.combinations(DEFAULT_RULES, i))

baseline_roles = ['free', 'cleaner', 'farmer']
BASELINE_SCENARIOS = []
for i in range(1, len(baseline_roles) + 1):
    BASELINE_SCENARIOS += list(itertools.combinations(baseline_roles, i))

TEST_SCENARIOS = [('learner',)]
for setting in BASELINE_SCENARIOS:
    TEST_SCENARIOS += [setting + ('learner',)]

# Make the dataframe and save it as a csv
start_time = time.time()

abs_count = 0
baseline_results = []
settings = {abs_count: [ROLE_SPRITE_DICT.keys()] + DEFAULT_RULES}

print()
print('*'*50)
print('STARTING BASELINE SCENARIOS')
print('*'*50)
print()

for i in range(len(BASELINE_SCENARIOS)):
    for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
      abs_count += 1
      roles = BASELINE_SCENARIOS[i]
      cur_settings, cur_result = main(roles=roles, 
                                      episodes=50, 
                                      num_iteration=i, 
                                      rules=rule_set, 
                                      create_video=True, 
                                      log_output=False)
      cur_df = pd.DataFrame.from_dict(cur_result)
      settings[abs_count] = cur_settings
      path = f"examples/results/baseline/abs{abs_count}_rule_set{rule_set_idx+1}-scenario{i+1}.csv"
      cur_df.to_csv(path_or_buf=path)
      print('='*50)
      print(f'BASELINE SCENARIO {i+1}/{len(BASELINE_SCENARIOS)} FOR RULE SET {rule_set_idx}/{len(RULE_COMBINATIONS)} COMPLETED')

print()
print('*'*50)
print('STARTING TEST SCENARIOS')
print('*'*50)
print()

test_results = []
for i in range(len(TEST_SCENARIOS)):
    for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
      abs_count += 1
      roles = TEST_SCENARIOS[i]
      cur_settings, cur_result = main(roles=roles, 
                                      episodes=150, 
                                      num_iteration=i, 
                                      rules=rule_set, 
                                      create_video=True, 
                                      log_output=False)
      cur_df = pd.DataFrame.from_dict(cur_result)
      settings[abs_count] = cur_settings
      path = f"examples/results/test/abs{abs_count}_rule_set{rule_set_idx+1}-scenario{i+1}.csv"
      cur_df.to_csv(path_or_buf=path)
      print('='*50)
      print(f'TEST SCENARIO {i+1}/{len(TEST_SCENARIOS)} FOR RULE SET {rule_set_idx+1}/{len(RULE_COMBINATIONS)} COMPLETED')

settings_df = pd.DataFrame.from_dict(settings)
settings_df.to_csv(path_or_buf="examples/results/test/settings.csv")
seconds = time.time() - start_time
hours = str(datetime.timedelta(seconds=seconds))
print('*'*50)
print('COMPLETED')
print(f"RUNTIME --- {hours} ---")
print('*'*50)