
from examples.eval_scripts.view_custom_model import main
import pandas as pd

# 1 get baseline
BASELINE_SCENARIOS = [
  #("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 0,
  #("cleaner",) * 0 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 0,
  #("cleaner",) * 0 + ("farmer",) * 0 + ('free',) * 1 + ('learner',) * 0,
  #("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 0,
  #("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 1 + ('learner',) * 0,
]

TEST_SCENARIOS = [
  ("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 1,
  #("cleaner",) * 0 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 1,
  #("cleaner",) * 0 + ("farmer",) * 0 + ('free',) * 1 + ('learner',) * 1,
  #("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 0 + ('learner',) * 1,
  #("cleaner",) * 1 + ("farmer",) * 1 + ('free',) * 1 + ('learner',) * 1,
  #("cleaner",) * 1 + ("farmer",) * 0 + ('free',) * 0 + ('learner',) * 1,
]

frames = []
baseline_results = []
for i in range(len(BASELINE_SCENARIOS)):
    roles = BASELINE_SCENARIOS[i]
    cur_result = main(roles=roles, episodes=20, num_iteration=i, create_video=False)
    cur_df = pd.DataFrame.from_dict(cur_result)
    frames.append(cur_df)

test_results = []
for i in range(len(TEST_SCENARIOS)):
    roles = TEST_SCENARIOS[i]
    cur_result = main(roles=roles, episodes=20, num_iteration=i, create_video=False)
    cur_df = pd.DataFrame.from_dict(cur_result)
    frames.append(cur_df)

path = "examples/results/run1.csv"
result = pd.concat(frames)
result.to_csv(path_or_buf=path)