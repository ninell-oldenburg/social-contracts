import itertools
import numpy as np
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
        "DIRT_FRACTION": ('>', list(np.arange(0,0.7,0.1)), '== 0'),
        "SINCE_AGENT_LAST_PAYED": ('>', list(np.arange(0,51,5)), '== 0'),
        "RIOTS": ("len(obs['RIOTS']) >= 1", None, "len(obs['RIOTS']) == 0"),
        "NUM_APPLES_AROUND": ('<', [0, 1, 2, 3, 4, 5, 6, 7, 8], '>'),
        "ORIENTATION": ('==', [0, 1, 2, 3], '!='),
        },
    "categorical": {
        "AGENT_LOOK": DEFAULT_LOOKS,
    }
}

DEFAULT_ACTIONS = [
    "MOVE_ACTION",
    "TURN_ACTION",
    "ZAP_ACTION",
    "CLEAN_ACTION",
    "CLAIM_ACTION",
    "EAT_ACTION",
    "PAY_ACTION"
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

    def generate_rules_of_length(self, target_length):
        obligations = []
        prohibitions = []

        for i in range(1, target_length+1):
            new_comb = list(itertools.combinations(self.features, i))
            for comb in new_comb:
                conditions = self.make_conditions(comb)
                prohibitions.extend(self.make_prohib_str(conditions))
                obligations.extend(self.make_oblig_str(comb, conditions))

        return obligations, prohibitions
            

    def make_prohib_str(self, conditions):
        prohibitions = [ProhibitionRule(condition, action) for action in self.actions \
                        for condition in conditions]

        return prohibitions
    
    def make_oblig_str(self, rule_elements, conditions):
        goals = self.make_conditions(rule_elements, is_goal=True)
        obligations = [ObligationRule(condition + f' and {look}', goal) for \
                       condition, goal in zip(conditions, goals) for look in self.looks]

        return obligations
    
    def make_conditions(self, rule_elements, is_goal=False):
        conditions = self.make_first_conditions(rule_elements, is_goal)
        for i, element in enumerate(rule_elements):
            if i != 0:
                for j in range(len(conditions)):
                    conditions = self.make_further_conditions(j, element, conditions, is_goal)
           
        return conditions
    
    def make_first_conditions(self, rule_elements, is_goal: bool):
        conditions = []
        element = rule_elements[0]

        if element in self.bools:
            conditions.append(''.join(f"obs['{element}']"))
        
        elif element in self.discretes:
            properties = self.discretes[element]
            compare_sign = properties[2] if is_goal else properties[0]
            if properties[1] == None:
                conditions.append(''.join(compare_sign))
            else:
                if not compare_sign == '== 0':
                    for param in properties[1]:
                        conditions.append(''.join(f"obs['{element}'] {compare_sign} {param}"))
                else: # RIOTS
                    conditions.append(''.join(f"obs['{element}'] {compare_sign}"))

        return conditions
        
    def make_further_conditions(self, i, element, conditions, is_goal: bool):

        for condition in conditions:
            if element in condition:
                return conditions

        if element in self.bools:
            if not is_goal:
                conditions[i] = conditions[i] + ' and ' + ''.join(f"obs['{element}']")
            else:
                conditions[i] = conditions[i] + ' and ' + ''.join(f"not obs['{element}']")
        
        elif element in self.discretes:
            properties = self.discretes[element]
            compare_sign = properties[2] if is_goal else properties[0]
            if properties[1] == None:
                conditions[i] = conditions[i] + " and " + "".join(compare_sign)
            else:
                if not compare_sign == '== 0':
                    for param in properties[1]:
                        conditions.append(conditions[i] + ' and ' + ''.join(f"obs['{element}'] {compare_sign} {param}"))
                    conditions.remove(conditions[i])
                else: # RIOTS
                    conditions[i] = conditions[i] + ' and ' + ''.join(f"obs['{element}'] {compare_sign}")
        
        return conditions


if __name__ == "__main__":
    generator = RuleGenerator()
    target_length = 2
    generated_rules = generator.generate_rules_of_length(target_length)
    for rule in generated_rules:
        # continue
        print(rule.make_str_repr())