import tensorflow as tf

Adagrad = lambda learning_rate : tf.train.AdagradOptimizer(learning_rate, name = "adagrad")
ADAM = lambda learning_rate : tf.train.AdamOptimizer(learning_rate, name = "adam")
Gradient_Descent = lambda learning_rate : tf.train.GradientDescentOptimizer(learning_rate, name = "gradient_descent")
RMS_prop = lambda learning_rate : tf.train.RMSPropOptimizer(learning_rate, name = "rms_prop")