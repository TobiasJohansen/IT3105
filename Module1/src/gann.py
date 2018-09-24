import numpy as np
import os
import tensorflow as tf
import time

# Turn off tensorflow information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class GANN():
    def __init__(self, network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, 
                 initial_weight_range, optimizer, casemanager):
        
        self.network_dimensions = network_dimensions
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.initial_weight_range = initial_weight_range
        self.optimizer = optimizer
        self.casemanager = casemanager

        self.build_network()
        self.configure_training()
    
    def build_network(self):
        self.input = tf.placeholder(tf.float64, shape = (None, self.network_dimensions[0]), name = "input-layer")
        self.modules = []
        input = self.input
        for i, dimension in enumerate(self.network_dimensions[1:-1]):
            self.modules.append(self.GANNModule(self.initial_weight_range, input, dimension, "hidden-layer-" + str(i + 1), 
                                                self.hidden_activation_function))
            input = self.modules[i].output
        self.modules.append(self.GANNModule(self.initial_weight_range, input, self.network_dimensions[-1], "output-layer", 
                                            self.output_activation_function))
        self.output = self.modules[-1].output

    def configure_training(self):
        self.target = tf.placeholder(tf.float64, shape = (None, self.network_dimensions[-1]), name = "target")
        self.error = self.cost_function(self.target, self.output)
        self.optimizer = self.optimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.error)

    def do_training(self, steps, minibatch_size, validation_interval):
        
        print("\n== TRAINING ==")        
        
        # Validation cases
        validation_inputs, validation_targets, validation_targets_as_ints = self.casemanager.validation_cases

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step_nr in range(1, steps + 1):
            
            # Train
            inputs, targets = self.casemanager.get_minibatch(minibatch_size)
            feeder = {self.input: inputs, self.target: targets}
            sess.run(self.trainer, feeder)

            # Consider validation
            if step_nr % validation_interval == 0:
                feeder = {self.input: validation_inputs, self.target: validation_targets}
                predictions = sess.run(self.output, feeder)
                top_k = sess.run(tf.nn.in_top_k(predictions, validation_targets_as_ints, 1))
                print("Validation test after {0} minibatches: {1:.2f}%"
                        .format(step_nr, 100.0 * np.sum(top_k) / len(validation_inputs)))

    class GANNModule():
        def __init__(self, initial_weight_range, input, dimension, name, activation_function):
            self.weights = tf.Variable(np.random.uniform(initial_weight_range[0], initial_weight_range[1], 
                                                         size = (input.get_shape().as_list()[1], dimension)), 
                                                         name = name + "_weights")
            self.bias = tf.Variable(np.random.uniform(initial_weight_range[0], initial_weight_range[1], 
                                                         size = dimension), 
                                                         name = name + "_biases")
            self.output = activation_function(tf.matmul(input, self.weights) + self.bias, name) 


