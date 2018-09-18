import src.system as system
import parameter_presets as preset

parameters = {
    "network_dimensions": [80],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.wine, "parameters": {}},
    "case_fraction": 1.0,
    "validation_fraction": 0.2,
    "validation_interval": 10,
    "test_fraction": 0.1,
    "minibatch_size": 50,
    "map_batch_size": 0,
    "steps": 10000,
    "map_layers": [],
    "map_dendrograms": [],
    "display_weights": [],
    "display_biases": []
}

system.run(**parameters)