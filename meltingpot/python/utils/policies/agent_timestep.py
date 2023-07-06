
import dm_env
import numpy as np

"""
calling the class:
x = AgentTimestep()
x.transfer_values(old_timestep)
x.add_non_physical_info(cur_timesteps)

TODO:
First AgentTimestep:
    Needs:
        property map, 
        normal map, 
        paying assignments (could be an array), 
        index, 
        look

    Updates: 
        map, 
        action_table
"""

class AgentTimestep():
    """
    Class for creating a timestep that adds relevant information
    from the computed dm_env.TimeStep to it's own class.
    """
    # TODO: what about self.observation['ALWAYS_PAYING_TO'] = 0
    def __init__(self) -> None:
        self.step_type = None
        self.reward = 0
        self.observation = {}
        self.parent_observation = {}
        self.index = None

    def get_obs(self):
        return self.observation
  
    def transfer_values(self, other_timestep):
        self.parent_observation = other_timestep.observation
        self.index = other_timestep.index
  
    def get_r(self):
        return self.reward
  
    def add_obs(self, obs_name: str, obs_val) -> None:
        self.observation[obs_name] = obs_val
        return
  
    def last(self):
        if self.step_type == dm_env.StepType.LAST:
            return True
        return False
  
    def add_non_physical_info(self, timestep: dm_env.TimeStep):
        """Adds Python information and adjusts coordinates."""
        self.step_type = timestep.step_type
        for obs_name, obs_val in timestep.observation.items():
            self.add_obs(obs_name=obs_name, obs_val=obs_val)

        x = self.observation['POSITION'][0] = timestep.observation['POSITION'][0]-1
        y = self.observation['POSITION'][1] = timestep.observation['POSITION'][1]-1
        self.observation['SURROUNDINGS'] = timestep.observation['MAP']
        self.observation['CUR_CELL_HAS_APPLE'] = True if self.observation['SURROUNDINGS'][x][y] == -3 else False
        self.make_territory_observation(x, y)
        self.make_action_observations(timestep, x, y)
        self.observation['NUM_APPLES_AROUND'] = self.get_apples()

    def make_action_observations(self, timestep: dm_env.TimeStep, x: int, y: int) -> None:
        """Compute everything that can be inferred from the action table."""
        self.observation['SINCE_AGENT_LAST_CLEANED'] = self.parent_observation['SINCE_AGENT_LAST_CLEANED'] + 1
        self.observation['SINCE_AGENT_LAST_PAYED'] = self.parent_observation['SINCE_AGENT_LAST_PAYED'] + 1
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
                    self.observation['SINGLE_AGENT_PAYED_LAST'] = 0
                    self.observation['INVENTORY'] -= 1

            if agent_action in [1, 2, 3, 4]:
                if self.observation['CUR_CELL_HAS_APPLE'] == True:
                    self.observation['INVENTORY'] += 1 # picked up apple

                    if self.observation['CUR_CELL_IS_FOREIGN_PROPERTY'] == True:
                        self.observation['AGENT_HAS_STOLEN'] = True
                        stolen_from = self.observation['PROPERTY'][x][y]
                        self.observation['STOLEN_RECORDS'][agent][stolen_from] = 1
                
        
    def get_apples(self):
        """Returns the sum of apples around a certain position."""
        sum = 0
        x, y = self.observation['POSITION'][0], self.observation['POSITION'][1]

        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if not self.exceeds_map(self.observation['WORLD.RGB'], i, j):
                    if not (i == x and j == y): # don't count target apple
                        if self.observation['SURROUNDINGS'][i][j] == -3:
                            sum += 1
        return sum

    def make_territory_observation(self, x: int, y: int) -> None:
        """
        Adds values for territory components to the observation dict.
            CUR_CELL_IS_FOREIGN_PROPERTY: True if current cell does not
                belong to current agent.
        """
        own_idx = self.index+1
        property_idx = int(self.observation['PROPERTY'][x][y])
        self.observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = False

        if property_idx != own_idx and property_idx != 0:
            self.observation['CUR_CELL_IS_FOREIGN_PROPERTY'] = True

    def exceeds_map(self, world_rgb: np.array, x: int, y: int) -> bool:
        """Returns True if current cell index exceeds game map."""
        x_max = world_rgb.shape[1] / 8
        y_max = world_rgb.shape[0] / 8
        if x < 0 or x >= x_max-1:
            return True
        if y < 0 or y >= y_max-1:
            return True