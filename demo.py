import numpy as np

from fuzzylogic import FuzzyInferenceSystem
from fuzzylogic.rules import disjunction, implication
from fuzzylogic.memfunc import trimf
from fuzzylogic.visual import plot_membership_functions, plot_aggregation


def main():

    fis = FuzzyInferenceSystem(
        quality=np.arange(0, 11, 1),
        service=np.arange(0, 11, 1),
        tip=np.arange(0, 26, 1)
    )

    # Define membership functions
    qual = trimf((0, 0, 5))
    qual_medium = trimf((0, 5, 10))
    qual_high = trimf((5, 10, 10))
    serv_low = trimf((0, 0, 5))
    serv_medium = trimf((0, 5, 10))
    serv_high = trimf((5, 10, 10))
    tip_low = trimf((0, 0, 13))
    tip_medium = trimf((0, 13, 25))
    tip_high = trimf((13, 25, 25))

    # Add membership functions to the FIS
    fis.add_membership_function('quality', 'low', qual)
    fis.add_membership_function('quality', 'medium', qual_medium)
    fis.add_membership_function('quality', 'high', qual_high)
    fis.add_membership_function('service', 'low', serv_low)
    fis.add_membership_function('service', 'medium', serv_medium)
    fis.add_membership_function('service', 'high', serv_high)
    fis.add_membership_function('tip', 'low', tip_low)
    fis.add_membership_function('tip', 'medium', tip_medium)
    fis.add_membership_function('tip', 'high', tip_high)

    plot_membership_functions(fis)

    # Rules
    # Rule 1: If the food is poor OR the service is poor, then the tip will be low
    rule_1 = disjunction([qual, serv_low], tip_low)

    # Rule 2: If the service is acceptable, then the tip will be medium
    rule_2 = implication(serv_medium, tip_medium)

    # Rule 3: If the food is good OR the service is good, then the tip will be high
    rule_3 = disjunction([qual_high, serv_high], tip_high)

    # Add rules to the FIS
    fis.add_rule(rule_1)
    fis.add_rule(rule_2)
    fis.add_rule(rule_3)

    # Inference
    quality = 6.5
    service = 9.8

    fis.infer(np.array([quality, service]), 'tip')
    plot_aggregation(fis, np.array([quality, service]), 'tip')


if __name__ == '__main__':
    main()
