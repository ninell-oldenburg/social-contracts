
import dm_env
import numpy as np

class AgentTimestep():
    def __init__(self) -> None:
        self.step_type = None
        self.reward = 0
        self.observation = {}
        self.goal = "apple"
        self.age = 0
        self.goal_count = 0
        self.MAX_LIFE_SPAN = 100

    def get_obs(self):
        return self.observation
    
    def get_r(self):
        return self.reward
    
    def add_obs(self, obs_name: str, obs_val) -> None:
        self.observation[obs_name] = obs_val
        return
    
    def last(self):
        if self.step_type == dm_env.StepType.LAST:
            return True
        return False
    
    def copy(self):
        new_ts = AgentTimestep()
        new_ts.step_type = self.step_type
        new_ts.reward = self.reward
        new_ts.observation = self.custom_deepcopy(self.observation)
        new_ts.goal = self.goal

        return new_ts

    def custom_deepcopy(self, old_obs):
        """Own copy implementation for time efficiency."""
        new_obs = {}
        for key, value in old_obs.items():
            if isinstance(value, np.ndarray):
                new_obs[key] = value.copy() if value.shape else value.item()
            else:
                new_obs[key] = value
        return new_obs

    def make_action_observations(self, timestep: dm_env.TimeStep, x: int, y: int) -> None:
        """Compute everything that can be inferred from the action table."""
        self.observation['SINCE_AGENT_LAST_CLEANED'] = self.parent_observation['SINCE_AGENT_LAST_CLEANED'] + 1
        self.observation['SINCE_AGENT_LAST_PAID'] = self.parent_observation['SINCE_AGENT_LAST_PAID'] + 1
        self.observation['TOTAL_NUM_CLEANERS'] = action_table[:][0].count(8)
        self.observation['STOLEN_RECORDS'] = self.parent_observation['STOLEN_RECORDS']
        self.observation['AGENT_HAS_STOLEN'] = False
        self.observation['INVENTORY'] = self.parent_observation['INVENTORY']
        
        num_agents = timestep.observation['ACTION_TABLE'].size()[0]
        action_table = timestep.observation['ACTION_TABLE']
        for agent in range(num_agents):
            agent_action = action_table[agent][0]

            if agent_action == 9: # claim
                self.observation['PROPERTY'][x][y] = agent

            if agent == self.index:
                if agent_action == 8: # clean
                    self.observation['SINGLE_AGENT_CLEANED_LAST'] = 0

                if agent_action == 10: # eat
                    self.observation['INVENTORY'] -= 1

                if agent_action == 11: # pay
                    self.observation['SINGLE_AGENT_PAID_LAST'] = 0
                    self.observation['INVENTORY'] -= 1

            if agent_action in [1, 2, 3, 4]:
                if self.observation['CUR_CELL_HAS_APPLE'] == True:
                    self.observation['INVENTORY'] += 1 # picked up apple

                    if self.observation['CUR_CELL_IS_FOREIGN_PROPERTY'] == True:
                        self.observation['AGENT_HAS_STOLEN'] = True
                        stolen_from = self.observation['PROPERTY'][x][y]
                        self.observation['STOLEN_RECORDS'][agent][stolen_from] = 1
                