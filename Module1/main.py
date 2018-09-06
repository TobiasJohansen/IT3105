from ann import ANN
import casemanager
import numpy as np
import tensorflow as tf
import tflowtools as tft

parameters = {
    "network_dimensions": [11, 8, 6],
    "hidden_activation_function": "relu",
    "output_activation_function": "softmax",
    "cost_function": "mse", 
    "learning_rate": 0.03,
    "initial_weight_range": { "upper_bound": 1, "lower_bound": 0},
    "optimizer": "gradient_descent",
    "data_source": "data/winequality_red.txt"
}

network = ANN(parameters)
network.run()