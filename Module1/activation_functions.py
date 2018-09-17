import tensorflow as tf

ReLU = lambda input, name : tf.nn.relu(input, name = name + "_relu")
Softmax = lambda input, name : tf.nn.softmax(input, name = name + "_softmax")