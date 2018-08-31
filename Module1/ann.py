import numpy as np
import tensorflow as tf

class ANN():
    
    activation_functions = {
        "relu": lambda features, name : tf.nn.relu(features, name),
        "softmax": lambda features, name : tf.nn.softmax(features, name)
    }

    def __init__(self, parameters):
        self.parameters = parameters

        network_dimensions = parameters["network_dimensions"]
        if len(network_dimensions) < 2 : raise Exception("Network must have at least two layers.")
        hidden_activation_function = parameters["hidden_activation_function"]
        output_activation_function = parameters["output_activation_function"]

        # All layers
        self.layers = []

        # Input layer
        self.layers.append({"layer": tf.placeholder(tf.float64, shape = (None, network_dimensions[0]), name = "input_layer")})
        
        # Hidden layers
        for i, dim in enumerate(network_dimensions[1:-1]):
            w = tf.Variable(np.random.uniform(-.1, .1, size = (network_dimensions[i], dim)), name = "w" + str(i + 1))
            b = tf.Variable(np.random.uniform(-.1, .1, size = dim), name = "b" + str(i + 1))
            hidden_layer = self.activation_functions[hidden_activation_function](tf.matmul(self.layers[i]["layer"], w) + b, "hidden_layer" + str(i + 1))
            self.layers.append({"w": w, "b": b, "layer": hidden_layer})

        # Output layer
        w = tf.Variable(np.random.uniform(-.1, .1, size = (network_dimensions[-2], network_dimensions[-1])), name = "w" + str(len(network_dimensions) - 1))
        b = tf.Variable(np.random.uniform(-.1, .1, size = network_dimensions[-1]), name = "b" + str(len(network_dimensions) - 1))
        output_layer = self.activation_functions[output_activation_function](tf.matmul(self.layers[-1]["layer"], w) + b, "output_layer")
        self.layers.append({"w": w, "b": b, "layer": output_layer})

        # Training
        self.target = tf.placeholder(tf.float64, shape = (None, network_dimensions[-1]), name = "Target")
        self.error = tf.reduce_mean(tf.square(self.target - self.layers[-1]["layer"]), name = "MSE")
        self.optimizer = tf.train.GradientDescentOptimizer(0.03)
        self.trainer = self.optimizer.minimize(self.error)

    def run(self, inputs, targets):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        feeder = {self.layers[0]["layer"]: inputs, self.target: targets}
        e = sess.run(self.error, feeder)

        while e > 0.02:
            result = sess.run([self.error, self.trainer], feeder)
            e = result[0]
            print("MSE:", e)
        print(sess.run(self.layers[-1]["layer"], feeder)[0])
