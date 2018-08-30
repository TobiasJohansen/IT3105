import tensorflow as tf
import tflowtools as tft

x = tf.Variable([[1,2], [3,4]])
y = tf.Variable(10)
z = x * y
session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run(z))