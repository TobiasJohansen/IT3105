import tensorflow as tf

relu = lambda input, name : tf.nn.relu(input, name = name + "_relu")
sigmoid = lambda input, name : tf.nn.sigmoid(input, name = name + "_sigmoid")
softmax = lambda input, name : tf.nn.softmax(input, name = name + "_softmax")
tanh = lambda input, name : tf.nn.tanh(input, name = name + "tanh")