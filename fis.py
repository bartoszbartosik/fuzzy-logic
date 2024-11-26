import numpy as np

from invertedpendulum import InvertedPendulum, plot

from fuzzylogic import FuzzyInferenceSystem
from fuzzylogic.rules import conjunction, disjunction, implication
from fuzzylogic.memfunc import trimf, trapmf
from fuzzylogic.visual import plot_membership_functions, plot_aggregation


def initialize_fis(s_0, plot=False):

    # --------------------- #
    #   U N I V E R S E S   #
    # --------------------- #

    fis = FuzzyInferenceSystem(
        angle=np.arange(0, 361, 0.1),
        ang_vel=np.arange(-6, 6, 0.1),
        force=np.arange(-100, 101, 0.1)
    )


    # ------------------------------------------- #
    #   M E M B E R S H I P   F U N C T I O N S   #
    # ------------------------------------------- #

    # Angle
    ang_small_left = trimf((359.5, 360, 360))
    ang_moderate_left = trimf((355, 357, 360))
    ang_big_left = trapmf((180, 180, 345, 360))

    ang_small_right = trimf((0, 0, 0.5))
    ang_moderate_right = trimf((0, 3, 5))
    ang_big_right = trapmf((0, 15, 180, 180))

    # Angular velocity
    ang_vel_small_left = trimf((-0.01, 0, 0))
    ang_vel_moderate_left = trimf((-0.5, -0.1, 0))
    ang_vel_big_left = trimf((-6, -6, 0))

    ang_vel_small_right = trimf((0, 0, 0.01))
    ang_vel_moderate_right = trimf((0, 0.1, 0.5))
    ang_vel_big_right = trimf((0, 6, 6))

    # Force
    force_small_left = trimf((-0.5, 0, 0))
    force_moderate_left = trimf((-20, -8, -1))
    force_big_left = trapmf((-100, -100, -30, -20))

    force_small_right = trimf((0, 0, 0.5))
    force_moderate_right = trimf((1, 8, 20))
    force_big_right = trapmf((20, 30, 100, 100))

    # Add membership functions to the FIS
    fis.add_membership_function('angle', 'small_left', ang_small_left)
    fis.add_membership_function('angle', 'moderate_left', ang_moderate_left)
    fis.add_membership_function('angle', 'big_left', ang_big_left)
    fis.add_membership_function('angle', 'small_right', ang_small_right)
    fis.add_membership_function('angle', 'moderate_right', ang_moderate_right)
    fis.add_membership_function('angle', 'big_right', ang_big_right)
    fis.add_membership_function('ang_vel', 'small_left', ang_vel_small_left)
    fis.add_membership_function('ang_vel', 'moderate_left', ang_vel_moderate_left)
    fis.add_membership_function('ang_vel', 'big_left', ang_vel_big_left)
    fis.add_membership_function('ang_vel', 'small_right', ang_vel_small_right)
    fis.add_membership_function('ang_vel', 'moderate_right', ang_vel_moderate_right)
    fis.add_membership_function('ang_vel', 'big_right', ang_vel_big_right)
    fis.add_membership_function('force', 'small_left', force_small_left)
    fis.add_membership_function('force', 'moderate_left', force_moderate_left)
    fis.add_membership_function('force', 'big_left', force_big_left)
    fis.add_membership_function('force', 'small_right', force_small_right)
    fis.add_membership_function('force', 'moderate_right', force_moderate_right)
    fis.add_membership_function('force', 'big_right', force_big_right)


    # ------------- #
    #   R U L E S   #
    # ------------- #

    # Rule 1: If angle is small_left and ang_vel is small_left then force is small_left
    rule_1 = conjunction([ang_small_left, ang_vel_small_left], force_moderate_left)
    # Rule 2: If angle is small_left and ang_vel is moderate_left then force is moderate_left
    rule_2 = conjunction([ang_small_left, ang_vel_moderate_left], force_big_left)
    # Rule 3: If angle is small_left and ang_vel is big_left then force is big_left
    rule_3 = conjunction([ang_small_left, ang_vel_big_left], force_big_left)

    # Rule 4: If angle is moderate_left and ang_vel is small_left then force is moderate_left
    rule_4 = conjunction([ang_moderate_left, ang_vel_small_left], force_big_left)
    # Rule 5: If angle is moderate_left and ang_vel is moderate_left then force is big_left
    rule_5 = conjunction([ang_moderate_left, ang_vel_moderate_left], force_big_left)
    # Rule 6: If angle is moderate_left and ang_vel is big_left then force is big_left
    rule_6 = conjunction([ang_moderate_left, ang_vel_big_left], force_big_left)

    # Rule 7: If angle is big_left and ang_vel is small_left then force is big_left
    rule_7 = conjunction([ang_big_left, ang_vel_small_left], force_big_left)
    # Rule 8: If angle is big_left and ang_vel is moderate_left then force is big_left
    rule_8 = conjunction([ang_big_left, ang_vel_moderate_left], force_big_left)
    # Rule 9: If angle is big_left and ang_vel is big_left then force is big_left
    rule_9 = conjunction([ang_big_left, ang_vel_big_left], force_big_left)

    # Rule 10: If angle is small_right and ang_vel is small_right then force is small_right
    rule_10 = conjunction([ang_small_right, ang_vel_small_right], force_moderate_right)
    # Rule 11: If angle is small_right and ang_vel is moderate_right then force is moderate_right
    rule_11 = conjunction([ang_small_right, ang_vel_moderate_right], force_big_right)
    # Rule 12: If angle is small_right and ang_vel is big_right then force is big_right
    rule_12 = conjunction([ang_small_right, ang_vel_big_right], force_big_right)

    # Rule 13: If angle is moderate_right and ang_vel is small_right then force is moderate_right
    rule_13 = conjunction([ang_moderate_right, ang_vel_small_right], force_big_right)
    # Rule 14: If angle is moderate_right and ang_vel is moderate_right then force is big_right
    rule_14 = conjunction([ang_moderate_right, ang_vel_moderate_right], force_big_right)
    # Rule 15: If angle is moderate_right and ang_vel is big_right then force is big_right
    rule_15 = conjunction([ang_moderate_right, ang_vel_big_right], force_big_right)

    # Rule 16: If angle is big_right and ang_vel is small_right then force is big_right
    rule_16 = conjunction([ang_big_right, ang_vel_small_right], force_big_right)
    # Rule 17: If angle is big_right and ang_vel is moderate_right then force is big_right
    rule_17 = conjunction([ang_big_right, ang_vel_moderate_right], force_big_right)
    # Rule 18: If angle is big_right and ang_vel is big_right then force is big_right
    rule_18 = conjunction([ang_big_right, ang_vel_big_right], force_big_right)

    # Rule 19: If angle is small_left and ang_vel is small_right then force is small_left
    rule_19 = conjunction([ang_small_left, ang_vel_small_right], force_moderate_right)
    # Rule 20: If angle is small_left and ang_vel is moderate_right then force is small_right
    rule_20 = conjunction([ang_small_left, ang_vel_moderate_right], force_moderate_left)
    # Rule 21: If angle is small_left and ang_vel is big_right then force is moderate_left
    rule_21 = conjunction([ang_small_left, ang_vel_big_right], force_big_right)

    # Rule 22: If angle is moderate_left and ang_vel is small_right then force is moderate_right
    rule_22 = conjunction([ang_moderate_left, ang_vel_small_right], force_big_left)
    # Rule 23: If angle is moderate_left and ang_vel is moderate_right then force is small_right
    rule_23 = conjunction([ang_moderate_left, ang_vel_moderate_right], force_small_right)
    # Rule 24: If angle is moderate_left and ang_vel is big_right then force is small_right
    rule_24 = conjunction([ang_moderate_left, ang_vel_big_right], force_small_right)

    # Rule 25: If angle is big_left and ang_vel is small_right then force is big_right
    rule_25 = conjunction([ang_big_left, ang_vel_small_right], force_big_left)
    # Rule 26: If angle is big_left and ang_vel is moderate_right then force is big_right
    rule_26 = conjunction([ang_big_left, ang_vel_moderate_right], force_big_left)
    # Rule 27: If angle is big_left and ang_vel is big_right then force is big_right
    rule_27 = conjunction([ang_big_left, ang_vel_big_right], force_big_left)

    # Rule 28: If angle is small_right and ang_vel is small_left then force is small_right
    rule_28 = conjunction([ang_small_right, ang_vel_small_left], force_small_left)
    # Rule 29: If angle is small_right and ang_vel is moderate_left then force is small_left
    rule_29 = conjunction([ang_small_right, ang_vel_moderate_left], force_small_right)
    # Rule 30: If angle is small_right and ang_vel is big_left then force is moderate_right
    rule_30 = conjunction([ang_small_right, ang_vel_big_left], force_moderate_left)

    # Rule 31: If angle is moderate_right and ang_vel is small_left then force is moderate_left
    rule_31 = conjunction([ang_moderate_right, ang_vel_small_left], force_moderate_right)
    # Rule 32: If angle is moderate_right and ang_vel is moderate_left then force is small_left
    rule_32 = conjunction([ang_moderate_right, ang_vel_moderate_left], force_small_right)
    # Rule 33: If angle is moderate_right and ang_vel is big_left then force is small_left
    rule_33 = conjunction([ang_moderate_right, ang_vel_big_left], force_small_right)

    # Rule 34: If angle is big_right and ang_vel is small_left then force is big_left
    rule_34 = conjunction([ang_big_right, ang_vel_small_left], force_big_right)
    # Rule 35: If angle is big_right and ang_vel is moderate_left then force is big_left
    rule_35 = conjunction([ang_big_right, ang_vel_moderate_left], force_big_right)
    # Rule 36: If angle is big_right and ang_vel is big_left then force is big_left
    rule_36 = conjunction([ang_big_right, ang_vel_big_left], force_big_right)

    # Add rules to the FIS
    fis.add_rule(rule_1)
    fis.add_rule(rule_2)
    fis.add_rule(rule_3)
    fis.add_rule(rule_4)
    fis.add_rule(rule_5)
    fis.add_rule(rule_6)
    fis.add_rule(rule_7)
    fis.add_rule(rule_8)
    fis.add_rule(rule_9)
    fis.add_rule(rule_10)
    fis.add_rule(rule_11)
    fis.add_rule(rule_12)
    fis.add_rule(rule_13)
    fis.add_rule(rule_14)
    fis.add_rule(rule_15)
    fis.add_rule(rule_16)
    fis.add_rule(rule_17)
    fis.add_rule(rule_18)
    fis.add_rule(rule_19)
    fis.add_rule(rule_20)
    fis.add_rule(rule_21)
    fis.add_rule(rule_22)
    fis.add_rule(rule_23)
    fis.add_rule(rule_24)
    fis.add_rule(rule_25)
    fis.add_rule(rule_26)
    fis.add_rule(rule_27)
    fis.add_rule(rule_28)
    fis.add_rule(rule_29)
    fis.add_rule(rule_30)
    fis.add_rule(rule_31)
    fis.add_rule(rule_32)
    fis.add_rule(rule_33)
    fis.add_rule(rule_34)
    fis.add_rule(rule_35)
    fis.add_rule(rule_36)

    # -----------------------

    if plot:
        fis.infer(np.array(s_0), 'force')

        # Plot
        plot_membership_functions(fis)
        plot_aggregation(fis, np.array(s_0), 'force')

    return fis
