from src.casemanager import Casemanager
from src.gann import GANN

# Imports to help user with parameters
from src.options import activation_functions, cost_functions, data_sources, optimizers

def run(network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, initial_weight_range, 
        optimizer, datasource, validation_fraction, validation_interval, test_fraction, minibatch_size, map_batch_size,
        steps, map_layers, map_dendrograms, display_weights, display_biases, case_fraction=1.0):
    
    casemanager = Casemanager(datasource, case_fraction, validation_fraction, test_fraction)
    gann = GANN(casemanager, network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, 
                 initial_weight_range, optimizer)

    error_log = gann.do_training(steps, minibatch_size, validation_interval)
    gann.do_testing()
    gann.visualize(error_log, map_batch_size, map_layers, map_dendrograms, display_weights, display_biases)