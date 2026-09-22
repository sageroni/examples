import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from optimization_test_functions import ad_campaign_profit



sample_input = np.array([0.9, 0.9, 1.0, 0.9])

# doing it the cheating way

def ob_fun_wrapper(input_value):
    return -ad_campaign_profit(input_value)
    # we minimize negative so we maximize

result =minimize(ob_fun_wrapper, x0 = sample_input, method='Nelder-Mead',
                 bounds = [(0, 1), (0, 1), (0, 1), (0, 1)])
#this will prevent it from geenrating solutions outside of these bounds

print(result)