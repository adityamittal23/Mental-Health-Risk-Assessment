import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_system():

    sleep = ctrl.Antecedent(np.arange(0, 11, 1), 'sleep')
    stress = ctrl.Antecedent(np.arange(0, 11, 1), 'stress')
    social = ctrl.Antecedent(np.arange(0, 11, 1), 'social')

    risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')

    # Membership functions
    sleep['low'] = fuzz.trimf(sleep.universe, [0, 0, 5])
    sleep['high'] = fuzz.trimf(sleep.universe, [5, 10, 10])

    stress['low'] = fuzz.trimf(stress.universe, [0, 0, 5])
    stress['high'] = fuzz.trimf(stress.universe, [5, 10, 10])

    social['low'] = fuzz.trimf(social.universe, [0, 0, 5])
    social['high'] = fuzz.trimf(social.universe, [5, 10, 10])

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 5])
    risk['high'] = fuzz.trimf(risk.universe, [5, 10, 10])

    # Rules (VERY IMPORTANT → ensure at least 1 always fires)
    rule1 = ctrl.Rule(sleep['low'] & stress['high'], risk['high'])
    rule2 = ctrl.Rule(sleep['high'] & stress['low'], risk['low'])
    rule3 = ctrl.Rule(social['low'], risk['high'])
    rule4 = ctrl.Rule(social['high'], risk['low'])

    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(system)


def get_fuzzy_score(simulator, sleep, stress, social):

    try:
        simulator.input['sleep'] = sleep
        simulator.input['stress'] = stress
        simulator.input['social'] = social

        simulator.compute()

        # SAFE RETURN
        if 'risk' in simulator.output:
            return simulator.output['risk']
        else:
            return 5  # fallback value

    except:
        return 5  # fallback if anything fails