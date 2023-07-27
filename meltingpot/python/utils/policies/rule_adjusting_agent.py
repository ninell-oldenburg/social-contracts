import dm_env

from collections import deque

import numpy as np

from copy import deepcopy

import random

from meltingpot.python.utils.substrates import shapes

from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule
from meltingpot.python.utils.policies.rule_learning_policy import RuleLearningPolicy
from meltingpot.python.utils.policies.agent_timestep import AgentTimestep
from meltingpot.python.utils.policies.lambda_rules import POTENTIAL_OBLIGATIONS, POTENTIAL_PROHIBITIONS
from meltingpot.python.utils.policies.lambda_rules import DEFAULT_PROHIBITIONS, DEFAULT_OBLIGATIONS

class RuleAdjustingPolicy(RuleLearningPolicy):

    def __init__(self,
                 env: dm_env.Environment,
                 player_idx: int,
                 # other_player_looks: list,
                 log_output: bool,
                 look: shapes,
                 num_players: int,
                 role: str = "learner",
                 potential_obligations: list = POTENTIAL_OBLIGATIONS,
                 potential_prohibitions: list = POTENTIAL_PROHIBITIONS,
                 active_prohibitions: list = DEFAULT_PROHIBITIONS,
                 active_obligations: list = DEFAULT_OBLIGATIONS,
                 selection_mode: str = "threshold",
                 threshold: int = 0.8,
                 ) -> None:
        
        # CALLING PARAMETERS
        self._index = player_idx
        self.role = role
        self.look = look
        self.log_output = log_output
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
        self.threshold = threshold

        # HYPERPARAMETER
        self.max_depth = 20
        self.compliance_cost = 0.1
        self.violation_cost = 0.3
        self.n_steps = 5
        self.gamma = 0.98
        self.n_rollouts = 10
        self.obligation_reward = 1.0
        self.initial_exp_r_cum = [30] * self.action_spec.num_values
        self.init_prior = 0.2
        
        # GLOBAL INITILIZATIONS
        self.history = deque(maxlen=10)
        self.payees = []
        self.riots = []
        self.hash_table = {}
        if self.role == 'farmer':
            self.payees = None
        # TODO condition on set of active rules
        self.V = {'apple': {}, 'clean': {}, 'pay': {}, 'zap': {}} # nested policy dict
        self.ts_start = None
        self.goal = None
        self.x_max = 15
        self.y_max = 15

        # non-physical info
        self.last_zapped = 0
        self.last_payed = 0
        self.last_cleaned = 0

        # self.player_looks = other_player_looks
        self.num_rules = len(self.potential_rules)
        self.rule_beliefs = np.array([self.threshold if rule in self.active_rules else self.init_prior for rule in self.potential_rules])
        self.nonself_active_obligations_count = np.array([dict() for _ in range(self.num_players)])
        self.others_history = deque(maxlen=10)
        self.history = deque(maxlen=10)

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
        
        self.relevant_keys = [
            'POSITION', 
            'ORIENTATION',
            'NUM_APPLES_AROUND',
            'POSITION_OTHERS',
            'INVENTORY',
            'CUR_CELL_IS_FOREIGN_PROPERTY', 
            'CUR_CELL_HAS_APPLE', 
            'AGENT_CLEANED'
        ]
    
    def step(self, 
           timestep: dm_env.TimeStep,
           actions: list
           ) -> list:
        """
        See base class.
        End of episode defined in dm_env.TimeStep.
        """

        self.x_max = len(timestep.observation[self._index]['WORLD.RGB'][1]) / 8
        self.y_max = len(timestep.observation[self._index]['WORLD.RGB'][0]) / 8

        ts_cur = self.add_non_physical_info(timestep=timestep, actions=actions, idx=self._index)
        self.ts_start = ts_cur

        # Check if any of the obligations are active
        self.current_obligation = None
        for obligation in self.obligations:
            cur_history = [ts[self._index].observation for ts in self.history]
            if obligation.holds_in_history(cur_history):
                self.current_obligation = obligation
                break
            
        self.set_goal()
                
        if self.log_output:
            print(f"player: {self._index} obligation active?: {self.current_obligation != None}")

        if not self.has_policy(ts_cur):
            self.rtdp(ts_cur)
        
        # return [self.get_best_act(ts_cur)]
        return self.get_optimal_path(ts_cur)
    
    def append_to_history(self, timestep_list: list) -> None:
        """Apoends a list of timesteps to the agent's history"""
        self.history.append(timestep_list)

    def update_beliefs(self, actions: list) -> None:
        """Update the beliefs of the rules based on the 
        observations and actions."""
        for player_idx in range(len(actions)):
            # Assumption: players are not updating on their own actions
            if not player_idx == self._index:
                # Compute the posterior of each rule
                self.compute_posterior(player_idx, actions[player_idx])

        # print(self.rule_beliefs)

    def maybe_mark_riot(self, player_idx, rule):
        """Saves the ones who are violating rules in the global riots variable."""
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
            if self.could_be_satisfied(rule, past_timestep):
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
        