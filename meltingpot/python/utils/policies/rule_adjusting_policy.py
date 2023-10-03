import dm_env

from collections import deque

import numpy as np
import random

from copy import deepcopy

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy
from meltingpot.python.utils.policies.agent_timestep import AgentTimestep
from meltingpot.python.utils.policies.rule_generation import RuleGenerator
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

generator = RuleGenerator()
POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS = generator.generate_rules_of_length(3)

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
DEFAULT_N_STEPS = 2
DEFAULT_N_ROLLOUTS = 2
DEFAULT_TAU = 1.0
DEFAULT_GAMMA = 0.9 # 0.85
DEFAULT_MAX_DEPTH = 20

# AGENT CLASS
DEFAULT_ACTION_COST = 0.01
DEFAULT_BLOCKING_COST = 0.1
DEFAULT_VIOLATION_COST = 5
DEFAULT_OBLIGATION_REWARD = 1
DEFAULT_APPLE_REWARD = 1
DEFAULT_SELECTION_MODE = "threshold"
DEFAULT_THRESHOLD = 0.95
DEFAULT_INIT_PRIOR = 0.05
DEFAULT_P_OBEY = 0.9
DEFAULT_OBLIGATION_DEPTH = 20
DEFAULT_AGE = 0
DEFAULT_MAX_LIFE_SPAN = 600

ROLE_SPRITE_DICT = {
   'free': shapes.CUTE_AVATAR,
   'cleaner': shapes.CUTE_AVATAR_W_SHORTS,
   'farmer': shapes.CUTE_AVATAR_W_FARMER_HAT,
   'learner': shapes.CUTE_AVATAR,
   }

INT_TO_ROLE = {
    0: 'free',
    1: 'cleaner',
    2: 'farmer',
    3: 'free'
}

ROLE_TO_INT = {
    'free': 0,
    'cleaner': 1,
    'farmer': 2,
    'learner': 3
}

class RuleAdjustingPolicy(RuleLearningPolicy):

    def __init__(self,
                env: dm_env.Environment,
                player_idx: int,
                log_output: bool,
                log_rule_prob_output: bool,
                log_weights: bool,
                look: int,
                num_players: int,
                role: int,
                potential_obligations: list = POTENTIAL_OBLIGATIONS,
                potential_prohibitions: list = POTENTIAL_PROHIBITIONS,
                active_prohibitions: list = DEFAULT_PROHIBITIONS,
                active_obligations: list = DEFAULT_OBLIGATIONS,
                selection_mode: str = DEFAULT_SELECTION_MODE,
                threshold: int = DEFAULT_THRESHOLD,
                max_depth: int = DEFAULT_MAX_DEPTH,
                violation_cost: float = DEFAULT_VIOLATION_COST,
                tau: float = DEFAULT_TAU, 
                n_steps: int = DEFAULT_N_STEPS, 
                gamma: float = DEFAULT_GAMMA,
                n_rollouts: int = DEFAULT_N_ROLLOUTS,
                obligation_reward: int = DEFAULT_OBLIGATION_REWARD,
                apple_reward: int = DEFAULT_APPLE_REWARD,
                element_blocking_cost: float = DEFAULT_BLOCKING_COST,
                default_action_cost: float = DEFAULT_ACTION_COST,
                init_prior: float = DEFAULT_INIT_PRIOR,
                p_obey: float = DEFAULT_P_OBEY,
                regrowth_probabilities: list = REGROWTH_PROBABILITIES,
                threshold_depletion: float = THRESHOLD_APPLE_DEPLETION,
                threshold_restoration: float = THRESHOLD_APPLE_RESTAURATION,
                max_apple_growth_rate: float = MAX_APPLE_GROWTH_RATE,
                dirt_spawn_prob: float = DIRT_SPAWN_PROB,
                is_learner: bool = False, 
                default_obligation_depth: int = DEFAULT_OBLIGATION_DEPTH,
                age: int = DEFAULT_AGE,
                MAX_LIFE_SPAN: int = DEFAULT_MAX_LIFE_SPAN,
                punish_cost: int = PENALTY_FOR_BEING_ZAPPED) -> None:
        
        # CALLING PARAMETERS
        self.py_index = player_idx
        self.lua_index = player_idx + 1
        self.role = role
        self.look = look
        self.log_output = log_output
        self.log_rule_prob_output = log_rule_prob_output
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
        self.intrinsic_violation_cost = violation_cost
        self.tau = tau
        self.n_steps = n_steps
        self.threshold = threshold
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.obligation_reward = obligation_reward
        self.apple_reward = apple_reward
        self.element_blocking_cost = element_blocking_cost
        self.max_depth = max_depth
        self.default_action_cost = default_action_cost
        self.init_prior = init_prior
        self.p_obey = p_obey
        self.is_learner = is_learner
        self.regrowth_probabilities = regrowth_probabilities
        self.num_regrowth_probs = len(self.regrowth_probabilities)
        self.threshold_depletion = threshold_depletion
        self.threshold_restoration = threshold_restoration
        self.max_apple_growth_rate = max_apple_growth_rate
        self.dirt_spawn_prob = dirt_spawn_prob
        self.max_obligation_depth = default_obligation_depth
        self.age = age
        self.MAX_LIFE_SPAN = MAX_LIFE_SPAN
        self.punish_cost = punish_cost
        
        # GLOBAL INITILIZATIONS
        self.history = deque(maxlen=10)
        self.step_counter = 0
        self.last_inventory = 0
        self.payees = None
        self.riots = []
        self.pos_all_possible_dirt = []
        self.pos_all_possible_apples = []
        self.current_obligations = deque()
        self.current_obligation = None
        self.hash_table = {}
        self.hash_count = {}
        self.q_value_log = {}
        goals = ['apple', 'clean', 'pay', 'zap']
        self.V = {goal: {} for goal in goals}
        self.V_wo_rules = {}
        self.all_bots = []
        self.ts_start = None
        self.goal = None
        self.x_max = 15
        self.y_max = 15
        self.dirt_fraction = 0.5
        self.interpolation = 0.5
        self.avg_steps_to_punishment = 50 / self.num_players

        # non-physical info
        self.last_zapped = 0
        self.last_paid = 0
        self.last_cleaned = 0
        self.old_pos = None
        self.freeze_counter = 0
        self.dirt_index = -3
        self.apple_index = -1
        self.water_index = -2

        # self.player_looks = other_player_looks
        self.num_rules = len(self.potential_rules)
        active_rule_strings = [rule.make_str_repr() for rule in self.active_rules]
        # if rules are coming in as active rules then they'll have a prior f self.threshold otherwise self.init_prior
        self.rule_beliefs = np.array([self.threshold if rule.make_str_repr() in active_rule_strings else self.init_prior for rule in self.potential_rules])
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
            'AGENT_PAID',
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'POSITION_OTHERS',
            'SINCE_AGENT_LAST_CLEANED',
            'SINCE_AGENT_LAST_PAID',
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
            'WATER_LOCATION', # maybe take out again
          ],
        'apple': [
            'AGENT_ATE',
            'CUR_CELL_HAS_APPLE',
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'INVENTORY',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'POSITION_OTHERS',
            'SINCE_AGENT_LAST_CLEANED',
            'SURROUNDINGS',
          ],
        'clean': [
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'DIRT_FRACTION',
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'POSITION_OTHERS',
            'SINCE_AGENT_LAST_CLEANED',
            'SURROUNDINGS',
          ],
          'pay': [
            'AGENT_PAID',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'POSITION_OTHERS',
            'SINCE_AGENT_LAST_PAID',
            'SURROUNDINGS',
          ],
        'zap': [
            'AGENT_ZAPPED',
            'CUR_CELL_HAS_APPLE', 
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'NUM_APPLES_AROUND',
            'ORIENTATION',
            'POSITION', 
            'POSITION_OTHERS',
            'SINCE_AGENT_LAST_ZAPPED',
            'SURROUNDINGS',
          ],
      }
        
        # if i'm a learner
        # look at other people's policy
        # get the best action according to boltzman
        #
        # prohibitions:
        #   look at best action without policy being true
        #   look at best action with policy being true
        # obligations:
        #   i don't know

    
    def step(self) -> list:
        """
        See base class.
        End of episode defined in dm_env.TimeStep.
        """

        ts_cur = self.history[-1][self.py_index]
        
        self.x_max = ts_cur.observation['WORLD.RGB'].shape[1] / 8
        self.y_max = ts_cur.observation['WORLD.RGB'].shape[0] / 8 - 5 # inventory display

        if self.log_rule_prob_output:
            print('='*50)
            print('CURRENT RULES')
            print('Obligations:')
            for rule in self.obligations:
                print(rule.make_str_repr())
            print()
            print('Prohibitions:')
            for rule in self.prohibitions:
                print(rule.make_str_repr())
            print('='*50)

        if ts_cur.step_type == dm_env.StepType.FIRST:
            self.pos_all_possible_apples = list(zip(*np.where(ts_cur.observation['SURROUNDINGS']== -1)))
            self.pos_all_possible_dirt = list(zip(*np.where(ts_cur.observation['SURROUNDINGS']== -3)))
            if self.role == 'farmer':
                if not type(ts_cur.observation['ALWAYS_PAYING_TO']) == np.int32:
                    self.payees = [i+1 for i, agent_one_hot in enumerate(ts_cur.observation['ALWAYS_PAYING_TO']) if agent_one_hot == 1]
                if self.payees == []:
                    self.payees = None

        else:
            if len(self.current_obligations) != 0:
                for obligation in list(self.current_obligations):
                    if obligation.satisfied(self.history[-2][self.py_index].observation):
                        self.current_obligations.remove(obligation)
            

        # Check if any of the obligations are active
        # self.current_obligation = None
        for obligation in self.obligations:
            cur_history = [ts[self.py_index].observation for ts in self.history]
            if obligation.holds_in_history(cur_history):
                if not obligation in self.current_obligations:
                    self.current_obligations.append(obligation)
                break
            
        self.set_goal()
        ts_cur.goal = self.goal

        self.handle_age()
                
        if self.log_output:
            print(f"player: {self.lua_index} obligation active?: {len(self.current_obligations) != 0}")

        if self.step_counter >= self.n_steps:
            self.rtdp(ts_cur)

        if not self.has_policy(ts_cur):
            self.rtdp(ts_cur)

        self.last_inventory = ts_cur.observation["INVENTORY"]

        return self.get_act(ts_cur, self.py_index, temp=0.0)
    
    def append_to_history(self, timestep_list: list) -> None:
        """Apoends a list of timesteps to the agent's history"""
        self.history.append(timestep_list)

    def set_all_bots(self, all_bots):
        self.all_bots = all_bots

    def handle_age(self):
        self.age += 1
        if self.age == self.MAX_LIFE_SPAN:
            self.freeze_counter = 5
        if self.freeze_counter == 1:
            self.age = 0
        self.freeze_counter = max(self.freeze_counter-1, 0)
    
    def role_exists_for_rule(self, rule) -> bool:
        for agent_history in self.history[-1]:
            agent_role = INT_TO_ROLE[agent_history.observation['AGENT_LOOK']]
            if agent_role in rule.make_str_repr():
                return True
        return False
    
    def get_look_for_value(self, value_to_find):
        for key, value in ROLE_SPRITE_DICT.items():
            if ''.join(value).encode('utf-8') == value_to_find:
                return key
        return None

    def update_beliefs(self, actions: list) -> None:
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for player_idx in range(len(actions)):
            # Assumption: players are not updating on their own actions
            if not player_idx == self.py_index:
                # Compute the posterior of each rule
                past_ts = self.history[-2][player_idx]
                this_ts = self.history[-1][player_idx]
                s_past_ts = self.hash_ts(past_ts)
                # get q values for active rules and as if there were no rules
                q_vals_no_rules = self.all_bots[player_idx].V_wo_rules[s_past_ts]
                # get boltzmann distribution for both q value vectors
                boltzmann_dis_no_rules = self.compute_boltzmann(q_vals_no_rules)
                action = actions[player_idx]
                self.compute_posterior(player_idx, action, this_ts, past_ts, boltzmann_dis_no_rules)

        # print(self.rule_beliefs)

    def maybe_mark_riot(self, player_idx, rule):
        """Saves the ones who are violating rules in the global riots variable."""
        if not player_idx == self.py_index:
            if rule in self.active_rules:
                self.riots.append(player_idx)

    def comp_oblig_llh(self, player_idx: int, rule: ObligationRule, action: int,
                       this_ts: AgentTimestep, past_ts: AgentTimestep, boltzmann_dis_no_rules: list) -> float:
        
        """
        Params: 
            player_idx: index of the player to be looked at
            rule:       rule to be investigated
            action:     action to be taken after "this ts"
            past_ts:    state of the environment beforehand
            this_ts:    last state of the environment
            """
        
        p_a_obs_no_rules = boltzmann_dis_no_rules[action]

        # unpack appearance, observation, position of the player
        # player_look = self.player_looks[player_idx]
        player_history = [self.history[i][player_idx].observation for i in range(len(self.history))]
        this_obs = this_ts.observation
        available = self.available_actions(past_ts.observation)

        q_vals_rule_is_active = np.full(self.action_spec.num_values , -1.0)
        for act in range(self.num_actions):
            ts_next =  self.all_bots[player_idx].env_step(past_ts, act, player_idx)
            s_next = self.all_bots[player_idx].init_process_next_ts(ts_next, player_idx)
            q_vals_rule_is_active[act], _, _ = self.all_bots[player_idx].get_bellmann_update(ts_next, s_next, act, available, past_ts, player_idx)
        
        boltzmann_dis_rule_is_active = self.compute_boltzmann(q_vals_rule_is_active)

        p_a_rule_is_active = boltzmann_dis_rule_is_active[action]

        if rule.holds_in_history(player_history[:-1]):
            if rule not in self.nonself_active_obligations_count[player_idx].keys():
                self.nonself_active_obligations_count[player_idx][rule] = 0
            else:
                self.nonself_active_obligations_count[player_idx][rule] += 1
        # if rule is not active remove it from the dict
        elif rule in self.nonself_active_obligations_count[player_idx].keys():
             self.nonself_active_obligations_count[player_idx].pop(rule, None)

        # Check if obligation rule is active
        rule_active_count = self.nonself_active_obligations_count[player_idx].get(rule, float('inf'))
        rule_is_active = rule_active_count <= self.max_obligation_depth

        if rule_is_active: # Obligation is active
            if self.could_be_satisfied(rule, past_ts, player_idx):
                if rule.satisfied(this_obs): # Agent obeyed the obligation
                    # P(obedient action | rule = true) = (1 * p_act_rule_is_active * p_obey) + (1 * p_act_np:rule_active * (1-p_obey))
                    p_action = self.p_obey * p_a_rule_is_active + p_a_obs_no_rules * (1-self.p_obey)
                    return np.log(p_action)
                else: # Agent disobeyed the obligation
                    # P(disobedient action | rule = true) = (0 * p_act_rule_is_active * p_obey) + (1 * p_act_np:rule_active * (1-p_obey))
                    self.maybe_mark_riot(player_idx, rule) # note rule violation
                    return np.log(p_a_obs_no_rules * (1-self.p_obey))
            else: # Obligation can't be satisfied
                return np.log(p_a_obs_no_rules)
        else: # Obligation is not active, or has expired
            return np.log(p_a_obs_no_rules)
        
    def comp_prohib_llh(self, player_idx: int, rule: ProhibitionRule, action: int,
                       this_ts: AgentTimestep, past_ts: AgentTimestep, boltzmann_dis_no_rules: list) -> float:

        """
        Params: 
            player_idx: index of the player to be looked at
            rule:       rule to be investigated
            action:     action to be taken after "this ts"
            past_ts:    state of the environment beforehand
            this_ts:    last state of the environment

            For this whole thing to work we need to do a bit of a headstand. The current 
            timestep displays the state of the world as it happened and does not account 
            potential futures (e.g. an apple will always be already collected and we're
            not able to see if that would have violated a rule). Therefore, we need to
            simulate what *would* have happened without the agent actually doing that action.
        
        """
        past_obs = past_ts.observation
        past_pos = past_obs['POSITION']

        # get a list of prohibited actions according to the ongoing rule
        prohib_actions = self.get_prohib_action(past_obs, rule, past_pos, player_idx)
        could_be_violated = True if len(prohib_actions) != 0 else False
        was_violated = True if action in prohib_actions else False
        available = set(range(self.num_actions)) - set(prohib_actions)
        
        q_vals_rule_is_active = np.full(self.action_spec.num_values , -1.0)
        for act in range(self.num_actions):
            ts_next =  self.all_bots[player_idx].env_step(past_ts, act, player_idx)
            s_next = self.all_bots[player_idx].init_process_next_ts(ts_next, player_idx)
            _, _, q_vals_rule_is_active[act]  = self.all_bots[player_idx].get_bellmann_update(ts_next, s_next, act, available, past_ts, player_idx)
        boltzmann_dis_rule_is_active = self.compute_boltzmann(q_vals_rule_is_active)

        for act in prohib_actions:
            boltzmann_dis_rule_is_active[act] = 0.0

        # get probabilities for observed action according to those distributions
        p_a_obs_no_rules = boltzmann_dis_no_rules[action]
        p_a_obs_rule_is_active = boltzmann_dis_rule_is_active[action]

        # always discount a violation
        # holds precondition means that with the step taken we have already violated the rule
        # what we want is something like: could be violated
        if could_be_violated:
            # P(disobedient action | rule = true) = 0 * p_action + p_action * (1-p_obey) 
            if was_violated:
               self.maybe_mark_riot(player_idx, rule)
            return np.log(p_a_obs_rule_is_active * self.p_obey + p_a_obs_no_rules * (1-self.p_obey))

        else:
           return np.log(p_a_obs_no_rules)
                        