import tensorflow as tf

relu = lambda input, name : tf.nn.relu(input, name = name + "_relu")
softmax = lambda input, name : tf.nn.softmax(input, name = name + "_softmax")