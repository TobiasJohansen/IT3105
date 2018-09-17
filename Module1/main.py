import system

parameters = {
    "network_dimensions": [80, 80],
    "hidden_activation_function": system.activation_functions.ReLU,
    "output_activation_function": system.activation_functions.Softmax,
    "cost_function": system.cost_functions.MSE,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.ADAM,
    #"datasource": {"function": system.data_sources.Parity, "parameters": {"num_bits": 5, "double": True}},
    "datasource": {"function": system.data_sources.Symmetry, "parameters": {"vlen": 10, "count": 5}},
    #"datasource": {"function": system.data_sources.Wine, "parameters": {}},
    "case_fraction": 1.0,
    "validation_fraction": 0.2,
    "validation_interval": 10,
    "test_fraction": 0.1,
    "minibatch_size": 50,
    "map_batch_size": 0,
    "steps": 100,
    "map_layers": [],
    "map_dendrograms": [],
    "display_weights": [],
    "display_biases": []
}

system.run(**parameters)