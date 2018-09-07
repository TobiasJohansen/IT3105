import casemanager
import numpy as np
import tensorflow as tf
import tflowtools as tft

class GANN():
    def __init__(self, dimensions):

        # Build the network
        self.modules = []
        input = None
        for i, dimension in enumerate(dimensions):
            self.modules.append(self.GANNModule(i, dimension, input))
            input = self.modules[i].output
        self.input = self.modules[0].output
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

        for i in range(epochs):
            _,r = sess.run([self.trainer, self.error], feeder)
            print(r)

    class GANNModule():
        def __init__(self, index, dimension, input):

            # If input layer
            if not index:
                self.output = tf.placeholder(tf.float64, shape = (None, dimension), name = "input")
            
            # If not input layer
            else:
                self.weights = tf.Variable(np.random.uniform(-.1, .1, size = (input.get_shape().as_list()[1], dimension)), name = "w" + str(index + 1))
                self.bias = tf.Variable(np.random.uniform(-.1, .1, size = dimension), name = "b" + str(index + 1))
                self.output = tf.nn.relu(tf.matmul(input, self.weights) + self.bias, "hidden_layer" + str(index + 1))

gann = GANN([11, 8, 6])
gann.run(100)
