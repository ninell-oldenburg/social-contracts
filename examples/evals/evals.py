from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS
from examples.view_custom_model import main

# 1 get baseline
BASELINE_SCENARIOS = [
  ("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 0,
  #("cleaner",) * 0 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 0,
  #("cleaner",) * 0 + ("farmer",) * 0 + ('free',) * 1 + ('learner',) * 0,
  #("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 0,
  #("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 1 + ('learner',) * 0,
]

TEST_SCENARIOS = [
  ("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 1,
  ("cleaner",) * 0 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 1,
  ("cleaner",) * 0 + ("farmer",) * 0 + ('free',) * 1 + ('learner',) * 1,
  ("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 1,
  ("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 1 + ('learner',) * 1,
  ("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 1,
]

results = []
for i in range(len(BASELINE_SCENARIOS)):
    roles = BASELINE_SCENARIOS[i]
    cur_result = main(roles=roles, episodes=50, num_iteration=i, create_video=False)
    results.append(cur_result)