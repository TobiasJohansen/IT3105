from ann import ANN
import numpy as np
import tensorflow as tf
import tflowtools as tft

parameters = {
    "network_dimensions": [32, 5, 32],
    "hidden_activation_function": "relu",
    "output_activation_function": "relu",
    "cost_function": "mse", 
    "learning_rate": 0.03,
    "initial_weight_range": { "upper_bound": 1, "lower_bound": 0},
    "optimizer": "gradient_descent"
}

network = ANN(parameters)

cases = np.array(tft.gen_all_one_hot_cases(32))
inputs = cases[:,0]
targets = cases[:,1]

network.run(inputs, targets)