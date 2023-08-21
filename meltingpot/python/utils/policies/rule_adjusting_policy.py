import dm_env

from collections import deque

import numpy as np
import random

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy
from meltingpot.python.utils.policies.agent_timestep import AgentTimestep
from meltingpot.python.utils.policies.rule_generation import RuleGenerator
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

generator = RuleGenerator()
POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS = generator.generate_rules_of_length(2)

# ENVIRONMENT COMPONENTS
APPLE_RESPAWN_RADIUS = 1.0
REGROWTH_PROBABILITIES = [0.02, 0.05, 0.07, 0.09]
OBSERVATION_RADIUS = 15 # defines radius that agents can observe
REMOVE_HIT_PLAYER = False # let them die when zapped
PENALTY_FOR_BEING_ZAPPED = 1 # violation cost
MAX_APPLE_GROWTH_RATE = 1
THRESHOLD_APPLE_DEPLETION = 0.7
THRESHOLD_APPLE_RESTAURATION = 0.1
DIRT_SPAWN_PROB = 0.2 # TODO to be unified

# RUN
DEFAULT_N_STEPS = 1
DEFAULT_N_ROLLOUTS = 2
DEFAULT_TAU = 0.0
# 0.99999 safe number: 0.99999
DEFAULT_GAMMA = 0.999999
DEFAULT_MAX_DEPTH = 20

# AGENT CLASS
DEFAULT_COMPLIANCE_COST = 0.001
DEFAULT_ACTION_COST = 0.001
DEFAULT_VIOLATION_COST = 0.5
DEFAULT_OBLIGATION_REWARD = 1
DEFAULT_APPLE_REWARD = 1
DEFAULT_COLLECT_APPLE_REWARD = 0.9
DEFAULT_SELECTION_MODE = "threshold"
DEFAULT_THRESHOLD = 0.8
DEFAULT_INIT_PRIOR = 0.2
DEFAULT_P_OBEY = 0.9

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_W_FARMER_HAT,
   'learner': shapes.CUTE_AVATAR,
   }

class RuleAdjustingPolicy(RuleLearningPolicy):

    def __init__(self,
                env: dm_env.Environment,
                player_idx: int,
                log_output: bool,
                log_weights: bool,
                look: shapes,
                num_players: int,
                role: str = "free",
                potential_obligations: list = POTENTIAL_OBLIGATIONS,
                potential_prohibitions: list = POTENTIAL_PROHIBITIONS,
                active_prohibitions: list = DEFAULT_PROHIBITIONS,
                active_obligations: list = DEFAULT_OBLIGATIONS,
                selection_mode: str = DEFAULT_SELECTION_MODE,
                threshold: int = DEFAULT_THRESHOLD,
                max_depth: int = DEFAULT_MAX_DEPTH,
                compliance_cost: float = DEFAULT_COMPLIANCE_COST,
                violation_cost: float = DEFAULT_VIOLATION_COST,
                tau: float = DEFAULT_TAU, 
                n_steps: int = DEFAULT_N_STEPS, 
                gamma: float = DEFAULT_GAMMA,
                n_rollouts: int = DEFAULT_N_ROLLOUTS,
                obligation_reward: int = DEFAULT_OBLIGATION_REWARD,
                apple_reward: int = DEFAULT_APPLE_REWARD,
                collect_apple_reward: float = DEFAULT_COLLECT_APPLE_REWARD,
                default_action_cost: float = DEFAULT_ACTION_COST,
                init_prior: float = DEFAULT_INIT_PRIOR,
                p_obey: float = DEFAULT_P_OBEY,
                regrowth_probabilities: list = REGROWTH_PROBABILITIES,
                threshold_depletion: float = THRESHOLD_APPLE_DEPLETION,
                threshold_restoration: float = THRESHOLD_APPLE_RESTAURATION,
                max_apple_growth_rate: float = MAX_APPLE_GROWTH_RATE,
                dirt_spawn_prob: float = DIRT_SPAWN_PROB) -> None:
        
        # CALLING PARAMETERS
        self.py_index = player_idx
        self.lua_index = player_idx + 1
        self.role = role
        self.look = look
        self.log_output = log_output
        self.log_weights = log_weights
        self.action_spec = env.action_spec()[0]
        self.selection_mode = selection_mode
        self.num_players = num_players
        self.num_actions = self.action_spec.num_values
        self.potential_obligations = potential_obligations
        self.potential_prohibitions = potential_prohibitions
        self.potential_rules = self.potential_prohibitions + self.potential_obligations
        self.prohibitions = active_prohibitions
        self.obligations = active_obligations
        self.active_rules = self.prohibitions + self.obligations

        # CONSTANTS
        self.max_depth = max_depth
        self.compliance_cost = compliance_cost
        self.violation_cost = violation_cost
        self.tau = tau
        self.n_steps = n_steps
        self.threshold = threshold
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.obligation_reward = obligation_reward
        self.apple_reward = apple_reward
        self.collect_apple_reward = collect_apple_reward
        self.max_depth = max_depth
        self.default_action_cost = default_action_cost
        self.init_prior = init_prior
        self.p_obey = p_obey
        self.regrowth_probabilities = regrowth_probabilities
        self.num_regrowth_probs = len(self.regrowth_probabilities)
        self.threshold_depletion = threshold_depletion
        self.threshold_restoration = threshold_restoration
        self.max_apple_growth_rate = max_apple_growth_rate
        self.dirt_spawn_prob = dirt_spawn_prob
        
        # GLOBAL INITILIZATIONS
        self.history = deque(maxlen=10)
        self.step_counter = 0
        self.last_inventory = 0
        self.payees = None
        self.riots = []
        self.pos_all_possible_dirt = []
        self.pos_all_possible_apples = []
        self.hash_table = {}
        self.hash_count = {}
        self.q_value_log = {}
        goals = ['apple', 'clean', 'pay', 'zap']
        self.V = {goal: {} for goal in goals}
        self.ts_start = None
        self.goal = None
        self.x_max = 15
        self.y_max = 15
        self.dirt_fraction = 0.5
        self.interpolation = 0.5

        # non-physical info
        self.last_zapped = 0
        self.last_payed = 0
        self.last_cleaned = 0
        self.old_pos = None

        # self.player_looks = other_player_looks
        self.num_rules = len(self.potential_rules)
        # if rules are coming in as active rules then they'll have a prior f self.threshold otherwise self.init_prior
        self.rule_beliefs = np.array([self.threshold if rule in self.active_rules else self.init_prior for rule in self.potential_rules])
        self.nonself_active_obligations_count = np.array([dict() for _ in range(self.num_players)])
        self.others_history = deque(maxlen=10)
        self.history = deque(maxlen=10)

        # print(f"SETTINGS\nmax_depth: {self.max_depth},\ntau: {self.tau},\nreward_scale_param: {self.reward_scale_param},\ngamma: {self.gamma}\n")

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
        
        # keep sorted
        self.relevant_keys = {
      'full': [
            'AGENT_ATE',
            'AGENT_CLAIMED',
            'AGENT_CLEANED',
            'AGENT_PAYED',
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_CLEANED',
            'SINCE_AGENT_LAST_PAYED',
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
            'WATER_LOCATION', # maybe take out again
            'POSITION_OTHERS',
          ],
        'apple': [
            'AGENT_ATE',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SURROUNDINGS',
          ],
        'clean': [
            'AGENT_CLEANED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_CLEANED',
            'SURROUNDINGS',
          ],
          'pay': [
            'AGENT_PAYED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_PAYED',
            'SURROUNDINGS',
            'POSITION_OTHERS',
          ],
        'zap': [
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
            'POSITION_OTHERS',
          ],
      }
    
    def step(self) -> list:
        """
        See base class.
        End of episode defined in dm_env.TimeStep.
        """

        self.x_max = self.history[-1][self.py_index].observation['WORLD.RGB'].shape[1] / 8
        self.y_max = self.history[-1][self.py_index].observation['WORLD.RGB'].shape[0] / 8 - 5 # inventory

        ts_cur = self.history[-1][self.py_index]
        self.ts_start = ts_cur
        self.ts_start.observation = self.custom_deepcopy(ts_cur.observation)

        if ts_cur.step_type == dm_env.StepType.FIRST:
            self.pos_all_possible_apples = list(zip(*np.where(ts_cur.observation['SURROUNDINGS']== -1)))
            self.pos_all_possible_dirt = list(zip(*np.where(ts_cur.observation['SURROUNDINGS']== -3)))
            if self.role == "farmer":
                if not type(ts_cur.observation['ALWAYS_PAYING_TO']) == np.int32:
                    self.payees = [i+1 for i, agent_one_hot in enumerate(ts_cur.observation['ALWAYS_PAYING_TO']) if agent_one_hot == 1]
                else:
                    self.payees = None

        # Check if any of the obligations are active
        self.current_obligation = None
        for obligation in self.obligations:
            cur_history = [ts[self.py_index].observation for ts in self.history]
            if obligation.holds_in_history(cur_history):
                self.current_obligation = obligation
                break
            
        self.set_goal()
                
        if self.log_output:
            print(f"player: {self.lua_index} obligation active?: {self.current_obligation != None}")

        if self.step_counter >= self.n_steps:
            self.rtdp(ts_cur)

        if not self.has_policy(self.ts_start):
            self.rtdp(ts_cur)

        self.last_inventory = ts_cur.observation["INVENTORY"]
        
        return self.get_best_act(self.ts_start)
    
    def append_to_history(self, timestep_list: list) -> None:
        """Apoends a list of timesteps to the agent's history"""
        self.history.append(timestep_list)

    def update_beliefs(self, actions: list) -> None:
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for player_idx in range(len(actions)):
            # Assumption: players are not updating on their own actions
            if not player_idx == self.py_index:
                # Compute the posterior of each rule
                self.compute_posterior(player_idx, actions[player_idx])

        # print(self.rule_beliefs)

    def maybe_mark_riot(self, player_idx, rule):
        """Saves the ones who are violating rules in the global riots variable."""
        if not player_idx == self.py_index:
            if rule in self.active_rules:
                self.riots.append(player_idx)

    def comp_oblig_llh(self, player_idx: int, rule: ObligationRule) -> float:

        # unpack appearance, observation, position of the player
        # player_look = self.player_looks[player_idx]
        player_history = [self.history[i][player_idx].observation for i in range(len(self.history))]
        next_obs = self.history[-1][player_idx].observation
        past_timestep = self.history[-2][player_idx] # needs to be ts for env_step(ts)

        if rule.holds_in_history(player_history[:-2]):
            if rule not in self.nonself_active_obligations_count[player_idx].keys():
                self.nonself_active_obligations_count[player_idx][rule] = 0
            else:
                self.nonself_active_obligations_count[player_idx][rule] += 1
        # if rule is not active remove it from the dict
        elif rule in self.nonself_active_obligations_count[player_idx].keys():
             self.nonself_active_obligations_count[player_idx].pop(rule, None)

        # Check if obligation rule is active
        rule_active_count = self.nonself_active_obligations_count[player_idx].get(rule, float('inf'))
        rule_is_active = rule_active_count <= self.max_depth

        if rule_is_active: # Obligation is active
            if self.could_be_satisfied(rule, past_timestep, player_idx):
                if rule.satisfied(next_obs): # Agent obeyed the obligation
                    # P(obedient action | rule = true) = (1 * p_obey) + 1/n_actions * (1-p_obey)
                    p_action = self.p_obey + (1-self.p_obey)/(self.num_actions)
                    return np.log(p_action)
                else: # Agent disobeyed the obligation
                    # P(disobedient action | rule = true) = (0 * p_obey) + 1/n_actions * (1-p_obey)  
                    p_action = (1-self.p_obey)/(self.num_actions)
                    # note rule violationg
                    self.maybe_mark_riot(player_idx, rule)
                    return np.log(p_action)
            else: # Obligation is not active, or has expired
                return np.log(1/(self.num_actions))
        else: # Obligation is not active, or has expired
            return np.log(1/(self.num_actions))
        
    def comp_prohib_llh(self, player_idx, rule, action) -> float:
    
        past_obs = self.history[-2][player_idx].observation
        past_pos = np.copy(past_obs['POSITION'])

        prohib_actions = self.get_prohib_action(past_obs, rule, past_pos)
        num_prohib_acts = len(prohib_actions)
        if rule.holds_precondition(past_obs):
            if action in prohib_actions: # violation
                # note rule violationg
                self.maybe_mark_riot(player_idx, rule)
                p_action = (1-self.p_obey)/(self.num_actions)
                return np.log(p_action)
            else: # action not prohibited
                p_action = self.p_obey + (1-self.p_obey)/(self.num_actions-num_prohib_acts)
                # p_action = 1/(self.num_actions-num_prohib_acts)
                return np.log(p_action)
        else: # precondition doesn't hold
            p_action = 1/(self.num_actions-num_prohib_acts)
            return np.log(p_action)
        
