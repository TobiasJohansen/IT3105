from casemanager import Casemanager
from gann import GANN

# Imports to help user with parameters
import activation_functions
import cost_functions
import data_sources
import optimizers

def run(network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, initial_weight_range, 
        optimizer, datasource, validation_fraction, validation_interval, test_fraction, minibatch_size, map_batch_size,
        steps, map_layers, map_dendrograms, display_weights, display_biases, case_fraction=1.0):
    
    casemanager = Casemanager(datasource, case_fraction, validation_fraction, test_fraction)
    
    network_dimensions.insert(0, casemanager.number_of_inputs)
    network_dimensions.append(casemanager.number_of_classes)
    
    gann = GANN(network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, 
                 initial_weight_range, optimizer, casemanager)
                 
    gann.run(steps)