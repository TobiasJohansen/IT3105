from casemanager import Casemanager
from gann import GANN
import data_sources

def run(network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, initial_weight_range, 
        optimizer, datasource, validation_fraction, validation_interval, test_fraction, minibatch_size, map_batch_size,
        steps, map_layers, map_dendrograms, display_weights, display_biases, case_fraction=1.0):
    casemanager = Casemanager(datasource, validation_fraction, test_fraction)
    gann = GANN(network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, 
                 initial_weight_range, optimizer, casemanager)
    gann.run(100)

run([11, 80, 80, 6], "relu", "softmax", "sigmoid_cross_entropy", 0.001, [-.1,.1], "adagrad", 
    [data_sources.Symmetry, 5, 1], 0.2, 10, 0.1, 50, 0, 100000, [], [], [], [])
