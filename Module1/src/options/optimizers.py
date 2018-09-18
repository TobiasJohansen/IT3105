import tensorflow as tf

adagrad = lambda learning_rate : tf.train.AdagradOptimizer(learning_rate, name = "adagrad")
adam = lambda learning_rate : tf.train.AdamOptimizer(learning_rate, name = "adam")
gradient_descent = lambda learning_rate : tf.train.GradientDescentOptimizer(learning_rate, name = "gradient_descent")
rms_prop = lambda learning_rate : tf.train.RMSPropOptimizer(learning_rate, name = "rms_prop")