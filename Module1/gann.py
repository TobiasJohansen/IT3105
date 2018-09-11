from casemanager import Casemanager
import numpy as np
import tensorflow as tf
import tflowtools as tft

activation_functions = {
    "relu": lambda input, name : tf.nn.relu(input, name=name),
    "softmax": lambda input, name : tf.nn.softmax(input, name=name)
}

cost_functions = {
    "cross-entropy": None,
    "mse": lambda output, target, name : tf.reduce_mean(tf.square(target - output), name = name)    
}

optimizers = {
    "adagrad": None,
    "adam": lambda learning_rate : tf.train.AdamOptimizer(learning_rate),
    "gradient_descent": None,
    "rms_prop": None    
}

class GANN():
    def __init__(self, network_dimensions, hidden_activation_function, output_activation_function, cost_function, learning_rate, 
                 initial_weight_range, optimizer, datasource, case_fraction=1.0):
        
        self.network_dimensions = network_dimensions
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.initial_weight_range = initial_weight_range
        self.optimizer = optimizer
        self.datasource = datasource
        self.case_fraction = case_fraction

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
        # Configure Training
        self.target = tf.placeholder(tf.float64, shape = (None, self.network_dimensions[-1]), name = "target")
        self.error = cost_functions[self.cost_function](self.output, self.target, name = self.cost_function)
        self.optimizer = optimizers[self.optimizer](self.learning_rate)
        self.trainer = self.optimizer.minimize(self.error)

    def run(self, epochs):
        casemanager = Casemanager(self.datasource[0], self.datasource[1])
        feeder = {self.input: casemanager.inputs, self.target: casemanager.one_hot_targets}

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for _ in range(epochs):
            _, r = sess.run([self.trainer, self.error], feeder)
            print(r)

    class GANNModule():
        def __init__(self, initial_weight_range, input, dimension, name, activation_function):
            self.weights = tf.Variable(np.random.uniform(initial_weight_range[0], initial_weight_range[1], 
                                                         size = (input.get_shape().as_list()[1], dimension)), 
                                                         name = name + "_weights")
            self.bias = tf.Variable(np.random.uniform(initial_weight_range[0], initial_weight_range[1], 
                                                         size = dimension), 
                                                         name = name + "_biases")
            self.output = activation_functions[activation_function](tf.matmul(input, self.weights) + self.bias, 
                                                                    name = name + "_" + activation_function)

gann = GANN([11, 80, 80, 6], "relu", "softmax", "mse", 0.001, [-.1,.1], "adam", ["data/winequality_red.txt", Casemanager.read_csv],
            )
gann.run(1000)
