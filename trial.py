import math
from glob import glob
import tensorflow as tf
import numpy as np

c = tf.Variable([[1.0, 2], [3, 4], [5, 6]])
b = tf.norm(c)
d = tf.zeros(c.shape)
d = tf.add(d,c)
d = tf.add(d,c)
d = tf.add(d,c)
d = tf.add(d,c)
d = tf.add(d,c)
a = np.zeros(c.shape)
r = tf.random_uniform([1,3])

# def cond_1():
# 	print("Yes")
# 	return 1

# def cond_2():
# 	print("No")
# 	return 0

x = tf.constant(2)
y = tf.constant(5)
def f1(): return tf.multiply(x, 17)
def f2(): return tf.add(y, 23)
r = tf.cond(tf.random_uniform([]) < 0.5, f1, f2)
d = tf.random_uniform([1])
# r = tf.cond(pred=tf.random_uniform([1])[0] < 0.5, true_fn=cond_1(), false_fn=cond_2())
with tf.Session() as sess:
	d = sess.run(d)
	# print(d)
	print(d)


    