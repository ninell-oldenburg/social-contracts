
from examples.eval_scripts.view_learning_model import main
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

from meltingpot.python.utils.policies.rule_generation import RuleGenerator
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

import itertools

"""def bayes_formula_obedience(prior):
    # Set the values for the conditional probability and the evidence
    # P(a) = P(a | r = 1) P(r = 1) + P(a | r = 0) P(r = 0)
    # P(r=1 | a) = P(a | r = 1) * P(r = 1) / P(a)
    llh = 0.083
    conditional = llh * prior
    marginal = conditional + ((1/12) * (1-prior))
    
    # Calculate the posterior probability using the Bayes formula
    posterior = conditional / marginal
    
    # Return the posterior probability
    return posterior

# Set the initial prior probability
prior = 0.77

# Generate a list of posterior probabilities using the Bayes formula
posterior_list = []
for i in range(100):
    posterior_list.append(prior)
    posterior = bayes_formula_obedience(prior)
    prior = posterior

# Plot the posterior probabilities as a function of the iteration number
plt.plot(range(len(posterior_list)), posterior_list)
plt.xlabel('Iteration Number')
plt.ylabel('Posterior Probability')
plt.title('Bayesian Inference')
plt.savefig(fname="update")
plt.show()"""

RULE_DICT = {
    'obs["CUR_CELL_HAS_APPLE"] and obs["NUM_APPLES_AROUND"] < 3 -> !MOVE_ACTION': "Too few apples prohibition",
    'obs["CUR_CELL_HAS_APPLE"] and obs["CUR_CELL_IS_FOREIGN_PROPERTY"] -> !MOVE_ACTION': "Don't steal prohibition",
    'obs["DIRT_FRACTION"] > 0.45 and obs["AGENT_LOOK"] == 0 -> obs["SINCE_AGENT_LAST_CLEANED"] == 0': "Egalitarian clean obligation",
    'obs["SINCE_AGENT_LAST_PAID"] > 30 and obs["AGENT_LOOK"] == 2 -> obs["SINCE_AGENT_LAST_PAID"] == 0': "Farmer pay obligation",
    'obs["DIRT_FRACTION"] > 0.3 and obs["AGENT_LOOK"] == 1 -> obs["SINCE_AGENT_LAST_CLEANED"] == 0': "Cleaner clean obligation"
}

generator = RuleGenerator()
POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS = generator.generate_rules_of_length(3)

EXPERIMENT_ROLES = ('cleaner',) * 1 + ('farmer',) * 1 + ('free',) * 1 + ('learner',) * 1
BASELINE_ROLES = ('cleaner',) * 1 + ('farmer',) * 1 + ('free',) * 1 + ('free',) * 1

BASLINE = []
for i in range(1, len(BASELINE_ROLES) + 1):
    new_comb = list(itertools.combinations(BASELINE_ROLES, i))
    for comb in new_comb:
        BASLINE.append(comb)

unique_sorted_tuples = set()
BASELINE_SCENARIOS = []
for tup in BASLINE:
    # Convert tuple to a sorted tuple
    sorted_tup = tuple(sorted(tup))
    # Check if this sorted tuple hasn't been seen before
    if sorted_tup not in unique_sorted_tuples:
        unique_sorted_tuples.add(sorted_tup)
        BASELINE_SCENARIOS.append(tup)

ALL_ROLE_COMB = []
for i in range(1, len(EXPERIMENT_ROLES) + 1):
    new_comb = list(itertools.combinations(EXPERIMENT_ROLES, i))
    for comb in new_comb:
        if not 'learner' in comb:
            continue
        ALL_ROLE_COMB.append(comb)

unique_sorted_tuples = set()
TEST_SCENARIOS = []
for tup in ALL_ROLE_COMB:
    # Convert tuple to a sorted tuple
    sorted_tup = tuple(sorted(tup))
    # Check if this sorted tuple hasn't been seen before
    if sorted_tup not in unique_sorted_tuples:
        unique_sorted_tuples.add(sorted_tup)
        TEST_SCENARIOS.append(tup)
        
DEFAULT_RULES = DEFAULT_PROHIBITIONS + DEFAULT_OBLIGATIONS
for rule in DEFAULT_RULES:
    if "RIOTS" in rule.make_str_repr():
        DEFAULT_RULES.remove(rule)

    # STR_RULES = [rule.make_str_repr() for rule in DEFAULT_RULES]
# Generate all possible combinations of the rules
RULE_COMBINATIONS = [] # include empty rule set
for i in range(0, len(DEFAULT_RULES) + 1):
    RULE_COMBINATIONS +=  list(itertools.combinations(DEFAULT_RULES, i))

start_time = time.time()

stats_relevance = 13

"""print(
   f'TEST_SCENARIOS: {TEST_SCENARIOS}\n'\
   f'BASELINE_SCENARIOS: {BASELINE_SCENARIOS}\n'\
)"""

RULE_STRINGS = []
for comb in RULE_COMBINATIONS:
    row = []
    for rule in comb:
        row.append(rule.make_str_repr())
    RULE_STRINGS.append(tuple(row))

# print(RULE_STRINGS)

RULE_NAMES = []
for comb in RULE_COMBINATIONS:
    row = []
    for rule in comb:
        row.append(RULE_DICT[rule.make_str_repr()])
    RULE_NAMES.append(tuple(row))

print()
print('*'*50)
print('STARTING BASELINE SCENARIOS')
print('*'*50)
print()

"""for k in range(stats_relevance):
  for i in range(len(BASELINE_SCENARIOS)):
      roles = BASELINE_SCENARIOS[i]
      cur_settings, cur_result = main(roles=roles, 
                                      episodes=300, 
                                      num_iteration=k, 
                                      rules=DEFAULT_RULES, 
                                      env_seed=k, 
                                      create_video=False, 
                                      log_output=False, 
                                      log_weights=False,
                                      save_csv=False,
                                      plot_q_vals=False
                                      )
      
      cur_df = pd.DataFrame.from_dict(cur_result)
      path = f'examples/results_selfish-learning/base/scenario{i+1}/trial{k+1}.csv'
      cur_df.to_csv(path_or_buf=path)
      print('='*50)
      print(f'ITERATION {k+1} BASELINE SCENARIO {i+1}/{len(BASELINE_SCENARIOS)} COMPLETED')

print()
print('*'*50)
print('STARTING TEST SCENARIOS')
print('*'*50)
print()

for k in range(stats_relevance):
  for i in range(len(TEST_SCENARIOS)):
      roles = TEST_SCENARIOS[i]
      cur_settings, cur_result = main(roles=roles, 
                                      episodes=300, 
                                      num_iteration=k, 
                                      rules=DEFAULT_RULES, 
                                      env_seed=k, 
                                      create_video=False, 
                                      log_output=False, 
                                      log_weights=False,
                                      save_csv=False,
                                      plot_q_vals=False,
                                    )
      cur_df = pd.DataFrame.from_dict(cur_result)
      path = f'examples/results_selfish-learning/test/scenario{i+1}/trial{k+1}.csv'
      cur_df.to_csv(path_or_buf=path)
      print('='*50)
      print(f'ITERATION {k+1} TEST SCENARIO {i+1}/{len(TEST_SCENARIOS)} COMPLETED')

print()
print('*'*50)
print('STARTING RULE BASELINE')
print('*'*50)
print()

for k in range(stats_relevance):
  for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
    cur_settings, cur_result = main(roles=BASELINE_ROLES,
                                    episodes=300, 
                                    num_iteration=k, 
                                    rules=rule_set, 
                                    env_seed=k, 
                                    create_video=False, 
                                    log_output=False, 
                                    log_weights=False,
                                    save_csv=False,
                                    plot_q_vals=False
                                    )
    
    cur_df = pd.DataFrame.from_dict(cur_result)
    path = f'examples/results_selfish-learning/rule_baseline/scenario{rule_set_idx+1}/trial{k+1}.csv'
    cur_df.to_csv(path_or_buf=path)
    print('='*50)
    print(f'ITERATION {k+1} RULE SET {rule_set_idx+1}/{len(RULE_COMBINATIONS)} COMPLETED')"""

print()
print('*'*50)
print('STARTING RULE TRIALS')
print('*'*50)
print()

for k in range(stats_relevance):
  for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
    cur_settings, cur_result = main(roles=EXPERIMENT_ROLES,
                                    episodes=300, 
                                    num_iteration=k, 
                                    rules=rule_set, 
                                    env_seed=k, 
                                    create_video=False, 
                                    log_output=False, 
                                    log_weights=False,
                                    save_csv=False,
                                    plot_q_vals=False
                                    )
    
    cur_df = pd.DataFrame.from_dict(cur_result)
    path = f'examples/results_selfish-learning/rule_trials/scenario{rule_set_idx+1}/trial{k+1}.csv'
    cur_df.to_csv(path_or_buf=path)
    print('='*50)
    print(f'ITERATION {k+1} RULE SET {rule_set_idx+1}/{len(RULE_COMBINATIONS)} COMPLETED')

# align length of df columns
for i in range(len(RULE_COMBINATIONS)):
  if i >= len(BASELINE_SCENARIOS):
      BASELINE_SCENARIOS.append('')
  if i >= len(TEST_SCENARIOS):
     TEST_SCENARIOS.append('')

# save settings as csv
settings = {
    'BASELINE_SCENARIOS': BASELINE_SCENARIOS, 
    'TEST_SCENARIOS': TEST_SCENARIOS,
    'RULE_COMBINATIONS': RULE_COMBINATIONS
    }
settings_df = pd.DataFrame.from_dict(settings)
settings_df.to_csv(path_or_buf='examples/results_selfish-learning/settings.csv')

settings_lambda = {
    'BASELINE_SCENARIOS': BASELINE_SCENARIOS, 
    'TEST_SCENARIOS': TEST_SCENARIOS,
    'RULE_COMBINATIONS': RULE_STRINGS,
}
settings_lambda_df = pd.DataFrame.from_dict(settings_lambda)
settings_lambda_df.to_csv(path_or_buf='examples/results_selfish-learning/settings_lambda.csv')

settings_names = {
    'BASELINE_SCENARIOS': BASELINE_SCENARIOS, 
    'TEST_SCENARIOS': TEST_SCENARIOS,
    'RULE_COMBINATIONS': RULE_NAMES,
}
settings_names_df = pd.DataFrame.from_dict(settings_names)
settings_names_df.to_csv(path_or_buf='examples/results_selfish-learning/settings_names.csv')

# TODO make lambda readable settings

seconds = time.time() - start_time
hours = str(datetime.timedelta(seconds=seconds))
print('*'*50)
print('COMPLETED')
print(f'RUNTIME --- {hours} ---')
print('*'*50)