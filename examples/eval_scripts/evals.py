
from examples.eval_scripts.view_custom_model import main, ROLE_SPRITE_DICT
import pandas as pd
import time
import datetime

from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

import itertools

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS

# Generate all possible combinations of the rules
RULE_COMBINATIONS = [] # include empty rule set
for i in range(0, len(DEFAULT_RULES) + 1):
    RULE_COMBINATIONS += list(itertools.combinations(DEFAULT_RULES, i))

print(len(RULE_COMBINATIONS))

baseline_roles = ['free', 'cleaner', 'farmer', 'learner']
BASELINE_SCENARIOS = [('free',), ('cleaner',), ('farmer',),]
TEST_SCENARIOS = []
for i in range(1, len(baseline_roles) + 1):
    new_comb = list(itertools.combinations(baseline_roles, i))
    for comb in new_comb:
      if 'learner' in comb:
         TEST_SCENARIOS.append(comb)
         if i > 1:
          lst = list(comb)
          idx = lst.index('learner')
          lst[idx] = 'free'
          new_comb = tuple(lst)
          BASELINE_SCENARIOS.append(new_comb)

print(len(BASELINE_SCENARIOS))
print(len(TEST_SCENARIOS))

# Make the dataframe and save it as a csv
start_time = time.time()

baseline_results = []
counter = 0
settings = {counter: [ROLE_SPRITE_DICT.keys()] + DEFAULT_RULES}

print()
print('*'*50)
print('STARTING BASELINE SCENARIOS')
print('*'*50)
print()

for i in range(len(BASELINE_SCENARIOS)):
    for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
      counter += 1
      roles = BASELINE_SCENARIOS[i]
      cur_settings, cur_result = main(roles=roles, 
                                      episodes=50, 
                                      num_iteration=i, 
                                      rules=rule_set, 
                                      create_video=True, 
                                      log_output=False)
      cur_df = pd.DataFrame.from_dict(cur_result)
      settings[counter] = cur_settings
      path = f"examples/results/baseline/rule_set{rule_set_idx+1}-scenario{i+1}.csv"
      cur_df.to_csv(path_or_buf=path)
      print('='*50)
      print(f'BASELINE SCENARIO {i+1}/{len(BASELINE_SCENARIOS)} FOR RULE SET {rule_set_idx+1}/{len(RULE_COMBINATIONS)} COMPLETED')

print()
print('*'*50)
print('STARTING TEST SCENARIOS')
print('*'*50)
print()

test_results = []
for i in range(len(TEST_SCENARIOS)):
    for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
      counter += 1
      roles = TEST_SCENARIOS[i]
      cur_settings, cur_result = main(roles=roles, 
                                      episodes=150, 
                                      num_iteration=i, 
                                      rules=rule_set, 
                                      create_video=True, 
                                      log_output=False)
      cur_df = pd.DataFrame.from_dict(cur_result)
      settings[counter] = cur_settings
      path = f"examples/results/test/rule_set{rule_set_idx+1}-scenario{i+1}.csv"
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