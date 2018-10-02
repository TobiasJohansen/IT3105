import src.system as system
import parameter_presets as preset

demo = {
    "network_dimensions": [10],
    "hidden_activation_function": system.activation_functions.relu,
    "output_activation_function": system.activation_functions.softmax,
    "cost_function": system.cost_functions.mse,
    "learning_rate": 0.001,
    "initial_weight_range": [-.1, .1],
    "optimizer": system.optimizers.adam,
    "datasource": {"function": system.data_sources.symmetry, "parameters": {"vlen": 5, "count": 100}},
    "case_fraction": 1.0,
    "validation_fraction": 0.1,
    "validation_interval": 100,
    "test_fraction": 0.1,
    "minibatch_size": 100,
    "map_batch_size": 10,
    "steps": 1000,
    "map_layers": [2],
    "map_dendrograms": [2],
    "display_weights": [2],
    "display_biases": [2]
}

system.run(**demo)