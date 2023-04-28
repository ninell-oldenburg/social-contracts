
from examples.eval_scripts.view_custom_model import main, ROLE_SPRITE_DICT
import pandas as pd
import time
import datetime

from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

import itertools

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

DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS
# Generate all possible combinations of the rules
RULE_COMBINATIONS = [] # include empty rule set
for i in range(0, len(DEFAULT_RULES) + 1):
    RULE_COMBINATIONS += list(itertools.combinations(DEFAULT_RULES, i))

start_time = time.time()

# save settings as csv
settings = {'BASELINE_SCENARIOS': BASELINE_SCENARIOS, 
            'TEST_SCENARIOS': TEST_SCENARIOS,
            'RULE_COMBINATIONS': RULE_COMBINATIONS}
settings_df = pd.DataFrame.from_dict(settings)
settings_df.to_csv(path_or_buf="examples/results/test/settings.csv")

print()
print('*'*50)
print('STARTING BASELINE SCENARIOS')
print('*'*50)
print()

for i in range(len(BASELINE_SCENARIOS)):
    roles = BASELINE_SCENARIOS[i]
    cur_settings, cur_result = main(roles=roles, 
                                    episodes=200, 
                                    num_iteration=i, 
                                    rules=DEFAULT_RULES, 
                                    create_video=True, 
                                    log_output=False)
    cur_df = pd.DataFrame.from_dict(cur_result)
    path = f"examples/results/baseline/scenario{i+1}.csv"
    cur_df.to_csv(path_or_buf=path)
    print('='*50)
    print(f'BASELINE SCENARIO {i+1}/{len(BASELINE_SCENARIOS)} COMPLETED')

print()
print('*'*50)
print('STARTING TEST SCENARIOS')
print('*'*50)
print()

test_results = []
for i in range(len(TEST_SCENARIOS)):
    roles = TEST_SCENARIOS[i]
    cur_settings, cur_result = main(roles=roles, 
                                    episodes=200, 
                                    num_iteration=i, 
                                    rules=DEFAULT_RULES, 
                                    create_video=True, 
                                    log_output=False)
    cur_df = pd.DataFrame.from_dict(cur_result)
    path = f"examples/results/test/scenario{i+1}.csv"
    cur_df.to_csv(path_or_buf=path)
    print('='*50)
    print(f'TEST SCENARIO {i+1}/{len(TEST_SCENARIOS)} COMPLETED')

print()
print('*'*50)
print('STARTING RULE TRIALS')
print('*'*50)
print()

for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
  cur_settings, cur_result = main(roles=roles, 
                                    episodes=50, 
                                    num_iteration=i, 
                                    rules=rule_set, 
                                    create_video=True, 
                                    log_output=False)
  cur_df = pd.DataFrame.from_dict(cur_result)
  path = f"examples/results/rules_trials/rule_set{rule_set_idx+1}.csv"
  cur_df.to_csv(path_or_buf=path)
  print('='*50)
  print(f'RULE SET {rule_set_idx+1}/{len(RULE_COMBINATIONS)} COMPLETED')

seconds = time.time() - start_time
hours = str(datetime.timedelta(seconds=seconds))
print('*'*50)
print('COMPLETED')
print(f"RUNTIME --- {hours} ---")
print('*'*50)