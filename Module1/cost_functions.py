import tensorflow as tf

MSE = lambda target, output : tf.losses.mean_squared_error(target, output)
Sigmoid_Cross_Entropy = lambda target, output : tf.losses.sigmoid_cross_entropy(target, output)
Softmax_Cross_Entropy = lambda target, output : tf.losses.softmax_cross_entropy(target, output)  