import torch.nn as nn
import torch.optim as optim
from statemanager import Statemanager
import rl_algoritm
import topp

best_params = {
    "hex_n": 5,
    "n_actual_games": 201,
    "n_search_games": 2000,
    "epsilon": 0.5,
    "learning_rate": 0.001,
    "hidden_layer_sizes": [64, 32],
    "hidden_activation_function": nn.ReLU,
    "optimizer": optim.Adam,
    "n_models": 4,
    "path": "models/topp/",
    "display_games": [],
    "g": 1000
}

test_params = {
    "hex_n": 5,
    "n_actual_games": 201,
    "n_search_games": 2000,
    "epsilon": 0.5,
    "learning_rate": 0.001,
    "hidden_layer_sizes": [64, 32],
    "hidden_activation_function": nn.ReLU,
    "optimizer": optim.Adam,
    "n_models": 4,
    "path": "models/test/",
    "display_games": [],
    "g": 1000
}

#rl_algoritm.train_models(**test_params)
topp.play_tournament(**test_params)
