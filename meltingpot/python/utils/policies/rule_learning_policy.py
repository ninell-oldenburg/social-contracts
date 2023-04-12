import dm_env

from collections import deque

import numpy as np

import ast

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_obeying_policy import RuleObeyingPolicy

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_HOLDING_PAINTBRUSH,
   'learner': shapes.CUTE_AVATAR,
   }

# PRECONDITIONS AND GOALS FOR OBLIGATIONS
cleaning_precondition_free = "lambda obs : obs['TOTAL_NUM_CLEANERS'] < 1"
cleaning_goal_free = "lambda obs : obs['TOTAL_NUM_CLEANERS'] >= 1"
payment_precondition_farmer = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] > 1"
payment_goal_farmer = "lambda obs : obs['SINCE_AGENT_LAST_PAYED'] <= 1"
cleaning_precondition_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] > 1"
cleaning_goal_cleaner = "lambda obs : obs['SINCE_AGENT_LAST_CLEANED'] <= 1"
payment_precondition_cleaner = "lambda obs : obs['SINCE_RECEIVED_LAST_PAYMENT'] > 1"
payment_goal_cleaner = "lambda obs : obs['SINCE_RECEIVED_LAST_PAYMENT'] <= 1"

POTENTIAL_OBLIGATIONS = [
  # clean the water if less than 1 agent is cleaning
  ObligationRule(cleaning_precondition_free, cleaning_goal_free),
  # If you're in the farmer role, pay cleaner with apples
  ObligationRule(payment_precondition_farmer, payment_goal_farmer, "farmer"),
  # If you're in the cleaner role, clean in a certain rhythm
  ObligationRule(cleaning_precondition_cleaner, cleaning_goal_cleaner, "cleaner"),
  # if you're a cleaner, wait until you've received a payment
  ObligationRule(payment_precondition_cleaner, payment_goal_cleaner, "cleaner")
]

learnables = {'num_apples_around': 3}

# PRECONDITIONS FOR PROHIBITIONS
harvest_apple_precondition = "lambda obs : obs['NUM_APPLES_AROUND'] < num_apples_around \
                                    and obs['CUR_CELL_HAS_APPLE']"
steal_from_forgein_cell_precondition = "lambda obs : obs['CUR_CELL_HAS_APPLE'] \
                                   and not obs['AGENT_HAS_STOLEN']"
  
POTENTIAL_PROHIBITIONS = [
  # don't go if <2 apples around
  ProhibitionRule(harvest_apple_precondition, 'MOVE_ACTION'),
  # don't go if it is foreign property and cell has apples 
  ProhibitionRule(steal_from_forgein_cell_precondition, 'MOVE_ACTION'),
]

class RuleLearningPolicy(RuleObeyingPolicy):
    def __init__(self,
                 env: dm_env.Environment,
                 player_idx: int,
                 player_looks: list,
                 role: str = "free",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS
                 ) -> None:
        
        self._index = player_idx
        self.role = role
        self._max_depth = 30
        self.action_spec = env.action_spec()[0]
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_rules = self.potential_obligations + self.potential_prohibitions
        self.obligations = []
        self.prohibitions = []
        self.current_obligation = None
        self.player_looks = player_looks
        self.num_total_agents = len(player_looks)
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([np.log(0.2)]*self.num_rules)
        self.nonself_active_obligations = np.array([set()]*self.num_total_agents)
        self.history = deque(maxlen=5)
        self.num_iterations = 0
        self.apple_count_list = []

        # move actions
        self.action_to_pos = [
            [[0,0],[0,-1],[0,1],[-1,0],[1,0]], # N
            [[0,0],[1,0],[-1,0],[0,-1],[0,1]], # E
            [[0,0],[0,1],[0,-1],[1,0],[-1,0]], # S
            [[0,0],[-1,0],[1,0],[0,1],[0,-1]], # W
            # N    # F    # SR  # B   # SL
          ]
        # turn actions
        self.action_to_orientation = [
            [3, 1], # N
            [0, 2], # E
            [1, 3], # S
            [2, 0], # W
          ]
        # non-move actions
        self.action_to_name = [
            "ZAP_ACTION",
            "CLEAN_ACTION",
            "CLAIM_ACTION",
            "EAT_ACTION",
            "PAY_ACTION"
          ]
        
    def update_beliefs(self, observations, other_agent_actions):
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for rule_index, rule in enumerate(self.potential_rules):
            # Compute the likelihood of the rule
            posterior = self.compute_posterior(rule_index,
                                               rule,
                                               observations, 
                                               other_agent_actions)
            
            self.rule_beliefs[rule_index] = posterior

    def compute_posterior(self, 
                          rule_index,
                          rule, 
                          observations, 
                          other_agent_actions):
        """Returns a posterior for a rule given an observation 
        and other agents' actions."""
        log_likelihood = np.log(0.5)
        pos_list = self.get_position_list(observations)
        has_learnable = False
        temp_precondition = ''

        for player_idx, action in enumerate(other_agent_actions):
        # check if the action violates the rule
            action_name = super().get_action_name(action)
            if not player_idx == self._index:
                # check if rule has a learnable
                precondition_ast = ast.parse(rule.precondition).body[0].value
                for node in ast.walk(precondition_ast):
                    if isinstance(node, ast.Name) and node.id in learnables.keys():
                        temp_precondition = rule.precondition
                        rule.precondition = rule.precondition.replace(node.id, str(learnables[node.id]))
                        has_learnable = True

                if isinstance(rule, ProhibitionRule):
                    # should only trigger if rule concerns behavior
                    if action_name == rule.prohibited_action:
                        pos = pos_list[player_idx]
                        x, y = pos[0], pos[1]
                        # update observation is only relevant for prohibitions
                        cur_player_obs = super().update_observation(observations, x, y)

                        if rule.holds(cur_player_obs, action_name):
                            log_likelihood *= 0.1  # rule is violated
                        else:
                            log_likelihood *= 0.9 # rule is obeyed
                            if has_learnable:
                                self.update_apple_param(cur_player_obs)

                elif isinstance(rule, ObligationRule):
                    player_role = self.get_role(player_idx)
                    if rule.holds_in_history(self.history, player_role):
                        if not rule in self.nonself_active_obligations[player_idx]:
                            # if we encounter an obligation precondition, save it
                            self.nonself_active_obligations[player_idx].add(rule)
                    if rule.satisfied(observations, player_role):
                        # if a previous precondition is fulfilled, likelihood increases
                        if rule in self.nonself_active_obligations:
                            log_likelihood *= 0.9
                        else:
                            log_likelihood *= 0.1

                # change back the inserted learnable param
                if has_learnable:
                    rule.precondition = temp_precondition

        prior = self.rule_beliefs[rule_index]
        marginal = prior * log_likelihood + (1 - prior) * (1 - log_likelihood)
        posterior = (prior * log_likelihood) / marginal

        return posterior
    
    def get_role(self, player_idx):
        for role in ROLE_SPRITE_DICT.keys():
            if self.player_looks[player_idx] == ROLE_SPRITE_DICT[role]:
                return role
        return "free"
    
    def update_apple_param(self, observations):
        cur_num_apples = observations['NUM_APPLES_AROUND']
        self.apple_count_list.append(cur_num_apples)
        learnables['num_apples_around'] = sum(self.apple_count_list) / len(self.apple_count_list)
    
    def sample_rules(self, threshold) -> None:
        """Updates the rules given a fixed number of highest priors."""
        obligations = []
        prohibitions = []
        for i, belief in enumerate(self.rule_beliefs):
            if np.exp(belief) > threshold:
                rule = self.potential_rules[i]
                if isinstance(rule, ObligationRule):
                    obligations.append(rule)
                elif isinstance(rule, ProhibitionRule):
                    prohibitions.append(rule)

        self.obligations = obligations
        self.prohibition = prohibitions
        
    def step(self,
             timestep: dm_env.TimeStep,
             other_agent_actions):
        """Use the learned rules to determine the actions of the agent."""
        self.num_iterations += 1

        self.history.append(timestep.observation)
        self.update_beliefs(timestep.observation, other_agent_actions)
        self.sample_rules(threshold=0.8)

        """print('='*50)
        print('CURRENT RULES')
        for x in self.obligations:
            print(x.precondition)
        for x in self.prohibitions:
            print(x.precondition)
        print('='*50)
        """
        # Check if any of the obligations are active
        self.current_obligation = None
        for obligation in self.obligations:
            if obligation.holds_in_history(self.history, self.role):
                self.current_obligation = obligation
            break

        print(f"player: {self._index} current_obligation active?: {self.current_obligation != None}")

        # use parent class to compute best step
        return super().a_star(timestep)
    
    def get_position_list(self, observation):
        position_list = [None]*self.num_total_agents
        for i in range(observation['SURROUNDINGS'].shape[0]):
            for j in range(observation['SURROUNDINGS'].shape[1]):
                if observation['SURROUNDINGS'][i][j] > 0: # agent encountered
                    agent_idx = observation['SURROUNDINGS'][i][j] 
                    position_list[agent_idx-1] = (i, j)

        return position_list