from examples.eval_scripts.view_intergen_model import main
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

from meltingpot.python.utils.policies.rule_generation import RuleGenerator
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from meltingpot.python.utils.policies.rule_adjusting_policy import DEFAULT_MAX_LIFE_SPAN

import itertools

generator = RuleGenerator()
POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS = generator.generate_rules_of_length(3)
DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS

ROLES = ('cleaner',) * 2 + ('farmer',) * 2 + ('free',) * 2
start_time = time.time()

stats_relevance = 5

print()
print('*'*50)
print('STARTING SCENARIOS')
print('*'*50)
print()

for k in range(stats_relevance):
    _, bot_dicts = main(roles=ROLES, 
                        episodes=DEFAULT_MAX_LIFE_SPAN*len(ROLES), 
                        num_iteration=k, 
                        rules=DEFAULT_RULES, 
                        env_seed=k, 
                        create_video=False, 
                        log_output=False, 
                        log_weights=False,
                        save_csv=False,
                        render=True,
                        )
      
    for j, cur_result in enumerate(bot_dicts):
        cur_df = pd.DataFrame.from_dict(cur_result)
        path = f'examples/results_intergen/max_life_span{DEFAULT_MAX_LIFE_SPAN}/bot{j+1}/trial{k+1}.csv'
        cur_df.to_csv(path_or_buf=path)

    print('='*50)
    print(f'ITERATION {k+1} BASELINE FOR MAX_LIFE_SPAN {DEFAULT_MAX_LIFE_SPAN} COMPLETED')

seconds = time.time() - start_time
hours = str(datetime.timedelta(seconds=seconds))
print('*'*50)
print('COMPLETED')
print(f'RUNTIME --- {hours} ---')
print('*'*50)