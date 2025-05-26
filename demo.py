import numpy as np

from fuzzylogic import FuzzyInferenceSystem
from fuzzylogic.rules import disjunction, implication, disj, Rule
from fuzzylogic.memfunc import trimf
from fuzzylogic.visual import plot_membership_functions, plot_aggregation
from fuzzylogic.structs import Universe, LinguisticVariable


def main():

    # Define universes
    quality = Universe('quality', np.arange(0, 11, 1))
    service = Universe('service', np.arange(0, 11, 1))
    tip = Universe('tip', np.arange(0, 26, 1))

    # Define membership functions
    quality.terms['low'] = trimf((0, 0, 5))
    quality.terms['medium'] = trimf((0, 5, 10))
    quality.terms['high'] = trimf((5, 10, 10))
    service.terms['low'] = trimf((0, 0, 5))
    service.terms['medium'] = trimf((0, 5, 10))
    service.terms['high'] = trimf((5, 10, 10))
    tip.terms['low'] = trimf((0, 0, 13))
    tip.terms['medium'] = trimf((0, 13, 25))
    tip.terms['high'] = trimf((13, 25, 25))

    # Create FIS
    fis = FuzzyInferenceSystem(quality, service, tip)

    # plot_membership_functions(fis)

    # Rules
    # Rule 1: If the food is poor OR the service is poor, then the tip will be low
    rule_1 = Rule([(quality, 'low'), (service, 'low')], disjunction, (tip, 'low'))
    rule_1([LinguisticVariable(6.5, quality.name), LinguisticVariable(9.8, service.name)])
    rule_1([6.5, 9.8], tip.domain)
    rule_1 = disjunction([quality.terms['low'], service.terms['low']], tip.terms['low'])

    # Rule 2: If the service is acceptable, then the tip will be medium
    rule_2 = implication(service.terms.medium, tip.terms.medium)

    # Rule 3: If the food is good OR the service is good, then the tip will be high
    rule_3 = disjunction([quality.terms.high, service.terms.high], tip.terms.high)

    # Add rules to the FIS
    fis.add_rule(rule_1)
    fis.add_rule(rule_2)
    fis.add_rule(rule_3)

    # Inference
    q = LinguisticVariable(6.5, quality)
    s = LinguisticVariable(9.8, service)

    # conj([(quality, 'low'), (service, 'low')], (tip, 'low'))([6.5, 9.8])

    fis.infer([q, s], tip)
    plot_aggregation(fis, np.array([q.value, s.value]), tip)



if __name__ == '__main__':
    main()
