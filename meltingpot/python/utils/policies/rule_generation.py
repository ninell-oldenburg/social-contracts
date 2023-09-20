import itertools
import numpy as np
from itertools import product
from meltingpot.python.utils.policies.ast_rules import ProhibitionRule, ObligationRule

DEFAULT_LOOKS = [
    "0",
    "1",
    "2"
]

DEFAULT_FEATURES = {
    "bool": [
        # "AGENT_CLEANED",
        "CUR_CELL_HAS_APPLE",
        "CUR_CELL_IS_FOREIGN_PROPERTY",
        # "AGENT_HAS_STOLEN"
        ],
    "discrete": {
        # 'KEY': (compare_condition, values, goal_condition)
        # "TOTAL_NUM_CLEANERS": ('<', [1, 2, 3, 4, 5], '>'),
        "DIRT_FRACTION": ('>', [0.375, 0.4, 0.425, 0.45, 0.475, 0.5], '== 0'),
        "SINCE_AGENT_LAST_PAID": ('>', list(np.arange(10,31,5)), '== 0'),
        "RIOTS": ('len(obs["RIOTS"]) > 0', None, 'len(obs["RIOTS"]) == 0'),
        "NUM_APPLES_AROUND": ('<', [1, 2, 3, 4, 5, 6, 7, 8], '>'),
        "ORIENTATION": ('==', [0, 1, 2, 3], '!='),
        },
    "categorical": {
        "AGENT_LOOK": DEFAULT_LOOKS,
    }
}

CELL_BASED_FEATURES = [
    "NUM_APPLES_AROUND",
    "CUR_CELL_HAS_APPLE",
    "CUR_CELL_IS_FOREIGN_PROPERTY"
]

ENV_BASED_FEATURES = [ 
    "DIRT_FRACTION",
]

AGENT_BASED_FEATURES = [
    "SINCE_AGENT_LAST_PAID",
    "ORIENTATION",
    "AGENT_LOOK"
]

DEFAULT_ACTIONS = [
    "MOVE_ACTION",
    #"TURN_ACTION",
    #"ZAP_ACTION",
    #'CLEAN_ACTION",
    #"CLAIM_ACTION",
    #"EAT_ACTION",
    #"PAY_ACTION"
]

class RuleGenerator():

    def __init__(self,
                 features: list = DEFAULT_FEATURES,
                 actions: list = DEFAULT_ACTIONS):
        self.actions = actions
        self.bools = features["bool"]
        self.discretes = features["discrete"]
        self.looks = features["categorical"]["AGENT_LOOK"]
        self.features = self.bools + list(self.discretes.keys())
        self.no_obligations = "CUR_CELL_HAS_APPLE", \
                            "CUR_CELL_IS_FOREIGN_PROPERTY", \
                            "NUM_APPLES_AROUND", \
                            "ORIENTATION"
        self.no_prohibitions = "SINCE_AGENT_LAST_PAID"
        self.higher_order = "DIRT_FRACTION"
        self.lower_order = "SINCE_AGENT_LAST_PAID"

    def generate_rules_of_length(self, target_length):
        obligations = []
        obligation_str = []
        prohibitions = []

        for i in range(1, target_length+1):
            new_comb = list(itertools.combinations(self.features, i))
            for comb in new_comb:
                # Check if all elements in comb are from the same list
                if all(elem in CELL_BASED_FEATURES for elem in comb) or \
                all(elem in ENV_BASED_FEATURES for elem in comb) or \
                all(elem in AGENT_BASED_FEATURES for elem in comb):
                    
                    # Check if 'NUM_APPLES_AROUND' is present, then 'CUR_CELL_HAS_APPLE' must also be present
                    if 'NUM_APPLES_AROUND' in comb and 'CUR_CELL_HAS_APPLE' not in comb:
                        continue

                    prohibitions.extend(self.make_prohib_str(comb))

                new_oblig = self.make_oblig_str(comb)
                for oblig in new_oblig:
                    if oblig.make_str_repr() not in obligation_str:
                        obligations.append(oblig)
                        obligation_str.append(oblig.make_str_repr())

        return obligations, prohibitions
            

    def make_prohib_str(self, rule_elements):
        if any(elem in self.no_prohibitions for elem in rule_elements):
            return []
        all_conditions = {}
        for i, element in enumerate(rule_elements):
            all_conditions[i] = self.get_conditions(element)

        string_combinations  = self.combine_entries(all_conditions)
        prohibitions = [ProhibitionRule(condition, action) for action in self.actions \
                        for condition in string_combinations]
        return prohibitions
    
    def make_oblig_str(self, rule_elements):
        # Remove elements that are in self.no_obligations
        if any(elem in self.no_obligations for elem in rule_elements):
            return []
        if any(elem in self.higher_order and elem2 in self.lower_order \
               for elem in rule_elements for elem2 in rule_elements):
            return []

        all_conditions = {}
        for i, element in enumerate(rule_elements):
            all_conditions[i] = self.get_conditions(element)
        
        string_combinations  = self.combine_entries(all_conditions)
            
        # Special case for DIRT_FRACTION
        if "DIRT_FRACTION" in rule_elements:
            goals = ['obs["SINCE_AGENT_LAST_CLEANED"] == 0']

        if "SINCE_AGENT_LAST_PAID" in rule_elements:
            goals = ['obs["SINCE_AGENT_LAST_PAID"] == 0']

        if 'RIOTS' in rule_elements:
            string_combinations = list(all_conditions[i])
            goals = ['len(obs["RIOTS"]) == 0']
            
        if 'RIOTS' in rule_elements:
            obligations = [ObligationRule(condition, goal) for condition in string_combinations for goal in goals]

        else:
            obligations = [ObligationRule(condition + f' and obs["AGENT_LOOK"] == {look}', goal) for \
                        condition in string_combinations for goal in goals for look in self.looks]
        
        return obligations
    
    def combine_entries(self, dct):
        combined_strings = []
        
        # Get all the lists from the dictionary
        lists = [dct[key] for key in dct]
        
        # Generate all combinations using itertools.product
        for combination in product(*lists):
            combined_string = ' and '.join(combination)
            combined_strings.append(combined_string)
        
        return combined_strings
    
    def get_conditions(self, elem_name) -> list:
        if elem_name in self.bools:
            return([f'obs["{elem_name}"]'])
        elif elem_name in self.discretes:
            properties = self.discretes[elem_name]
            compare_sign = properties[0]
            if properties[1] == None:
                return([compare_sign])
            else:
                return [(f'obs["{elem_name}"] {compare_sign} {param}') for param in properties[1]]

    def make_conditions(self, rule_elements):
        conditions = []
        for element in rule_elements:
            if element in self.bools:
                conditions.append(f'obs["{element}"]')
            elif element in self.discretes:
                compare_sign, values, _ = self.discretes[element]
                for value in values:
                    conditions.append(f'obs["{element}"] {compare_sign} {value}')

        return conditions
        
    def make_further_conditions(self, i, element, conditions, is_goal: bool):

        for condition in conditions: # no doubles
            if element in condition:
                return conditions

        if element in self.bools:
            if not is_goal:
                conditions[i] = conditions[i] + " and " + "".join(f'obs["{element}"]')
            else:
                conditions[i] = conditions[i] + ' and ' + ''.join(f'not obs["{element}"]')
        
        elif element in self.discretes:
            properties = self.discretes[element]
            compare_sign = properties[2] if is_goal else properties[0]
            if properties[1] == None:
                conditions[i] = conditions[i] + ' and ' + ''.join(compare_sign)
            else:
                if not compare_sign == '== 0':
                    for param in properties[1]:
                        conditions.append(conditions[i] + ' and ' + ''.join(f'obs["{element}"] {compare_sign} {param}'))
                    conditions.remove(conditions[i])
                else: # RIOTS
                    conditions[i] = conditions[i] + ' and ' + ''.join(f'obs["{element}"] {compare_sign}')
        
        return conditions


if __name__ == "__main__":
    generator = RuleGenerator()
    target_length = 2
    prohibs, obligs = generator.generate_rules_of_length(target_length)
    for rule in prohibs + obligs:
        # continue
        print("'" + rule.make_str_repr() + "',")