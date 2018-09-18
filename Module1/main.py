import src.system as system

parameters = {
    "network_dimensions": [10, 5],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.parity, "parameters": {"num_bits": 10}},
    #"datasource": {"function": system.data_sources.symmetry, "parameters": {"vlen": 10, "count": 5}},
    #"datasource": {"function": system.data_sources.Wine, "parameters": {}},
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