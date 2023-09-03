
from examples.eval_scripts.view_custom_model import main
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

generator = RuleGenerator()
POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS = generator.generate_rules_of_length(3)

DEFAULT_ROLES = ('cleaner',) * 1 + ('farmer',) * 1 + ('free',) * 1 + ('learner',) * 1
BASELINE_ROLES = ('free',) * 1 + ('cleaner',) * 1 + ('farmer',) * 1 + ('free',) * 1

baseline_roles = ['free', 'cleaner', 'farmer', 'free']
BASELINE_SCENARIOS = [('free',), ('cleaner',), ('farmer',), ('free',)]
ALL_ROLE_COMB = []
for i in range(1, len(baseline_roles) + 1):
    new_comb = list(itertools.combinations(baseline_roles, i))
    for comb in new_comb:
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
# STR_RULES = [rule.make_str_repr() for rule in DEFAULT_RULES]
# Generate all possible combinations of the rules
RULE_COMBINATIONS = [] # include empty rule set
for i in range(0, len(DEFAULT_RULES) + 1):
    RULE_COMBINATIONS +=  list(itertools.combinations(DEFAULT_RULES, i))

start_time = time.time()

stats_relevance = 1

print()
print('*'*50)
print('STARTING BASELINE SCENARIOS')
print('*'*50)
print()


for k in range(stats_relevance):
  for i in range(len(BASELINE_SCENARIOS)):
      roles = BASELINE_SCENARIOS[i]
      cur_settings, cur_result = main(roles=DEFAULT_ROLES, 
                                      episodes=300, 
                                      num_iteration=k, 
                                      rules=DEFAULT_RULES, 
                                      env_seed=k, 
                                      create_video=False, 
                                      log_output=False, 
                                      log_weights=False,
                                      save_csv=True,
                                      plot_q_vals=False,
                                      gamma=0.9999,
                                      tau=0.5,
                                      )

      cur_df = pd.DataFrame.from_dict(cur_result)
      path = f'examples/results/base/scenario{i+1}/trial{k+1}.csv'
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
                                      save_csv=True,
                                      plot_q_vals=False,
                                      gamma=0.9999,
                                      tau=0.5,
                                      )
      print(cur_result)
      cur_df = pd.DataFrame.from_dict(cur_result)
      path = f'examples/results/test/scenario{i+1}/trial{k+1}.csv'
      cur_df.to_csv(path_or_buf=path)
      print('='*50)
      print(f'ITERATION {k+1} TEST SCENARIO {i+1}/{len(TEST_SCENARIOS)} COMPLETED')

      
print()
print('*'*50)
print('STARTING RULE TRIALS')
print('*'*50)
print()

for k in range(stats_relevance):
  for rule_set_idx, rule_set in enumerate(RULE_COMBINATIONS):
    cur_settings, cur_result = main(roles=DEFAULT_ROLES, 
                                    episodes=300, 
                                    num_iteration=k, 
                                    rules=rule_set, 
                                    env_seed=k, 
                                    create_video=False, 
                                    log_output=False, 
                                    log_weights=False,
                                    save_csv=True,
                                    plot_q_vals=False,
                                    threshold_init_prior=0.8,
                                    learner_init_prior=0.2,
                                    gamma=0.99999,
                                    tau=0.5)
    
    cur_df = pd.DataFrame.from_dict(cur_result)
    path = f'examples/results/rule_baseline/scenario{rule_set_idx+1}/trial{k+1}.csv'
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
settings = {'BASELINE_SCENARIOS': BASELINE_SCENARIOS, 
            'TEST_SCENARIOS': TEST_SCENARIOS,
            'RULE_COMBINATIONS': RULE_COMBINATIONS}
settings_df = pd.DataFrame.from_dict(settings)
settings_df.to_csv(path_or_buf='examples/results/settings.csv')

seconds = time.time() - start_time
hours = str(datetime.timedelta(seconds=seconds))
print('*'*50)
print('COMPLETED')
print(f'RUNTIME --- {hours} ---')
print('*'*50)