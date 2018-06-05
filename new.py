import numpy as np
import tensorflow as tf


n_features = 5
m_examples = 69

# The input examples. The i-th row contains the i-th example. 
X_true = np.loadtxt("x2.txt").astype(np.float32).reshape(-1,n_features)
# These are our labeled "results". 
y_true = np.loadtxt("y.txt").astype(np.float32).reshape(-1,1)
 
# The constant
b_true = np.random.rand(1) * 10

print "w = [%s]" % ', '.join(['%.2f' % x for x in w_true])
print "b = %.2f" % b_true[0]

# Noise
noise = np.random.rand(m_examples, 1) / 100.0


# Placeholder that is fed input data.
X_in = tf.placeholder(tf.float32, [None, n_features], "X_in")

# The model:multi input linear regression we assume y = X_in * w + b
w = tf.Variable(tf.random_normal((n_features, 1)), name="w")
b = tf.Variable(tf.constant(0.1, shape=[]), name="b")
# 
h = tf.add(tf.matmul(X_in, w), 0, name="h")

# Placeholder that is fed observed results.
y_in = tf.placeholder(tf.float32, [None, 1], "y_in")

# loss function
loss_op = tf.reduce_mean(tf.square(y_in - h), name="loss")

# loss_op = tf.reduce_sum(tf.pow(y_in - , 2))/(2 * m_examples)
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        sess.run(train_op, feed_dict={
          X_in: X_true, 
          y_in: y_true
        })
    w_computed = sess.run(w)
    b_computed = sess.run(b)
    print w_computed
    print b_computed


print "w computed [%s]" % ', '.join(['%.5f' % x for x in w_computed.flatten()])
print "w actual   [%s]" % ', '.join(['%.5f' % x for x in w_true.flatten()])
print "b computed %.3f" % b_computed
print "b actual  %.3f" % b_true[0]