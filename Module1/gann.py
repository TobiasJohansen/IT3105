import casemanager
import numpy as np
import tensorflow as tf
import tflowtools as tft

class GANN():
    def __init__(self, dimensions):

        # Build the network
        self.input = tf.placeholder(tf.float64, shape = (None, dimensions[0]), name = "input_layer")
        self.modules = []
        input = self.input
        for i, dimension in enumerate(dimensions[1:-1]):
            self.modules.append(self.GANNModule(input, dimension, "hidden_layer_" + str(i + 1)))
            input = self.modules[i].output
        self.modules.append(self.GANNModule(input, dimensions[-1], "output_layer"))
        self.output = self.modules[-1].output

        # Training
        self.target = tf.placeholder(tf.float64, shape = (None, dimensions[-1]), name = "target")
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name = "mse")
        self.optimizer = tf.train.AdamOptimizer()
        self.trainer = self.optimizer.minimize(self.error)
    
    def run(self, epochs):
        cases = casemanager.get_cases("data/winequality_red.txt")
        inputs = cases[:,:-1]
        targets = [int(target) - 3 for target in cases[:,-1]]
        one_hot_targets = [tft.int_to_one_hot(target, 6) for target in targets]

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        feeder = {self.input: inputs, self.target: one_hot_targets}

        for _ in range(epochs):
            _, r = sess.run([self.trainer, self.error], feeder)
            print(r)

    class GANNModule():
        def __init__(self, input, dimension, name):
            self.weights = tf.Variable(np.random.uniform(-.1, .1, size = (input.get_shape().as_list()[1], dimension)), name = name + "_weights")
            self.bias = tf.Variable(np.random.uniform(-.1, .1, size = dimension), name = name + "_biases")
            self.output = tf.nn.relu(tf.matmul(input, self.weights) + self.bias, name = name + "_output")

gann = GANN([11, 80, 80, 6])
gann.run(1000)
