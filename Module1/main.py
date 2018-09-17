import system

parameters = {
    "network_dimensions": [11, 80, 80, 6],
    "hidden_activation_function": system.activation_functions.ReLU,
    "output_activation_function": system.activation_functions.Softmax,
    "cost_function": system.cost_functions.Sigmoid_Cross_Entropy,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.Adagrad,
    "datasource": {"function": system.data_sources.Parity, "parameters": {"num_bits": 5}},
    "case_fraction": 1.0,
    "validation_fraction": 0.2,
    "validation_interval": 10,
    "test_fraction": 0.1,
    "minibatch_size": 50,
    "map_batch_size": 0,
    "steps": 100000,
    "map_layers": [],
    "map_dendrograms": [],
    "display_weights": [],
    "display_biases": []
}

system.run(**parameters)