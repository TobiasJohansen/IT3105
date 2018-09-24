import src.system as system

parity = {
    "network_dimensions": [10, 5],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.parity, "parameters": {"num_bits": 10}},
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

symmetry = {
    "network_dimensions": [10, 5],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.symmetry, "parameters": {"vlen": 10, "count": 5}},
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

wine = {
    "network_dimensions": [100, 20],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.wine, "parameters": {}},
    "case_fraction": 1.0,
    "validation_fraction": 0.1,
    "validation_interval": 1000,
    "test_fraction": 0.1,
    "minibatch_size": 100,
    "map_batch_size": 0,
    "steps": 100000,
    "map_layers": [],
    "map_dendrograms": [],
    "display_weights": [],
    "display_biases": []
}

mnist = {
    "network_dimensions": [100, 80, 60, 40, 20, 10],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.mnist, "parameters": {}},
    "case_fraction": 0.1,
    "validation_fraction": 0.1,
    "validation_interval": 1000,
    "test_fraction": 0.1,
    "minibatch_size": 50,
    "map_batch_size": 0,
    "steps": 10000,
    "map_layers": [],
    "map_dendrograms": [],
    "display_weights": [],
    "display_biases": []
}