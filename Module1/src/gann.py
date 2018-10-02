import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tools.tflowtools as tft
import time

# Turn off tensorflow information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class GANN():
    
    def __init__(self, casemanager, network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, 
                 initial_weight_range, optimizer):
        
        self.casemanager = casemanager
        network_dimensions.insert(0, casemanager.number_of_features)
        network_dimensions.append(casemanager.number_of_classes)
        self.network_dimensions = network_dimensions
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.initial_weight_range = initial_weight_range
        self.optimizer = optimizer
        

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
        self.trainer = self.optimizer.minimize(self.error, name = "backprop")

    def evaluate(self, cases):
        inputs, targets, targets_as_ints = cases
        feeder = {self.input: inputs, self.target: targets}
        predictions, error = self.sess.run([self.output, self.error], feeder)
        top_k = self.sess.run(tf.nn.in_top_k(predictions, targets_as_ints, 1))
        return 100.0 * np.sum(top_k) / len(inputs), error

    def do_training(self, steps, minibatch_size, validation_interval):
        
        error_log = {"training": {"step_nr": [], "error": []} , "validation": {"step_nr": [], "error": []}}
        
        print("\n== TRAINING ==")        
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for step_nr in range(1, steps + 1):
            
            # Train
            inputs, targets, _ = self.casemanager.get_minibatch(minibatch_size)
            feeder = {self.input: inputs, self.target: targets}
            _, error = self.sess.run([self.trainer, self.error], feeder)
            error_log["training"]["step_nr"].append(step_nr)
            error_log["training"]["error"].append(error)

            # Consider validation
            if step_nr % validation_interval == 0:
                accuracy, error = self.evaluate(self.casemanager.get_validation_cases())
                print("Validation test after {0} minibatches: {1:.2f}%".format(step_nr, accuracy))
                error_log["validation"]["step_nr"].append(step_nr)
                error_log["validation"]["error"].append(error)
        
        return error_log

    def do_testing(self):
        print("\n== TESTING ==")        
        print("Train data: {0:.2f}%".format(self.evaluate(self.casemanager.get_train_cases())[0]))
        print("Test data: {0:.2f}%".format(self.evaluate(self.casemanager.get_test_cases())[0]))

    def plot_error(self, error_log):
        plt.figure()
        ax = plt.gca()
        plt.plot(error_log["training"]["step_nr"], error_log["training"]["error"], label="Training Error")
        plt.plot(error_log["validation"]["step_nr"], error_log["validation"]["error"], label="Validation Error")
        plt.legend()
        ax.set_title("Error")

    def vals_to_grab(self, map_layers, map_dendrograms, display_weights, display_biases):
        grabvals = [[],[],[],[]]
        
        # Map Layers
        for layer in map_layers:
            if layer == 0:
                activation = self.input
            else:
                activation = self.modules[layer - 1].output
            grabvals[0].append(activation)
        
        # Dendrograms
        for layer in map_dendrograms:
            if layer == 0:
                activation = self.input
            else:
                activation = self.modules[layer - 1].output
            grabvals[1].append(activation)

        # Display Weights
        for layer in display_weights:
            grabvals[2].append(self.modules[layer - 1].weights)
        
        # Display Biases
        for layer in display_biases:
            grabvals[3].append(self.modules[layer - 1].bias)

        return grabvals

    def plot_map_layers(self, map_layers, activations):
        if len(activations) > 0:
            for i, activation in enumerate(activations):
                tft.hinton_plot(np.array(activation), title="Hinton Plot, Layer - " + str(map_layers[i]))

    def plot_dendrograms(self, map_dendrograms, activations, inputs, targets_as_ints):
        if len(activations) > 0:
            non_bit = any(any(isinstance(number, float) for number in input) for input in inputs)
            if non_bit or len(inputs[0]) > 6:
                labels = [target_as_int for target_as_int in targets_as_ints]
            else:
                labels = [tft.bits_to_str(input) for input in inputs]
            for i, activation in enumerate(activations):
                plt.figure()
                tft.dendrogram(activation, labels, title="Dendrogram, Layer - " + str(map_dendrograms[i]))
    
    def plot_weights(self, display_weights, activations):
        if len(activations) > 0:
            for i, activation in enumerate(activations):
                tft.display_matrix(np.array(activation), title="Weights, Layer - " + str(display_weights[i]))

    def plot_biases(self, display_biases, activations):
        if len(activations) > 0:
            for i, activation in enumerate(activations):
                tft.display_matrix(np.array([activation]), title="Biases, Layer - " + str(display_biases[i]))

    def visualize(self, error_log, map_batch_size, map_layers, map_dendrograms, display_weights, display_biases):
        
        self.plot_error(error_log)

        if map_batch_size > 0:
                    
            grabvals = self.vals_to_grab(map_layers, map_dendrograms, display_weights, display_biases)
            inputs, _, targets_as_ints = self.casemanager.get_minibatch(map_batch_size)
            result = self.sess.run(grabvals, {self.input: inputs}) 

            print("\n== MAP BATCH ==")
            [print(input) for input in inputs]

            self.plot_map_layers(map_layers, result[0])
            self.plot_dendrograms(map_dendrograms, result[1], inputs, targets_as_ints)
            self.plot_weights(display_weights, result[2])
            self.plot_biases(display_biases, result[3])

        plt.show()

    class GANNModule():

        def __init__(self, initial_weight_range, input, dimension, name, activation_function):
            
            input_size = input.get_shape().as_list()[1]
            
            if initial_weight_range == "scaled":
                v = 1 / np.sqrt((input_size)) 
                initial_weight_range = [-v, v]
                
            self.weights = tf.Variable(np.random.uniform(initial_weight_range[0], initial_weight_range[1], 
                                                         size = (input_size, dimension)), 
                                                         name = name + "_weights")
            self.bias = tf.Variable(np.random.uniform(initial_weight_range[0], initial_weight_range[1], 
                                                         size = dimension), 
                                                         name = name + "_biases")
            self.output = activation_function(tf.matmul(input, self.weights) + self.bias, name) 