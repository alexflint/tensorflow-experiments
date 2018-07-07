import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

def main():
    with tf.summary.FileWriter('out/events') as writer:
        writer.add_graph(tf.get_default_graph())

    # features and labels
    xs = tf.constant([[1.], [2.], [3.], [4.]])
    ys_true = tf.constant([[2.], [4.], [6.], [8.]])

    # linear model
    ys_predicted = tf.layers.dense(xs, units=1)

    # loss
    loss = tf.losses.mean_squared_error(ys_true, ys_predicted)

    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # initialization
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    # run
    s = tf.Session()
    s.run((var_init, table_init))
    for i in range(20):
        _, cur_loss = s.run((train, loss))
        print(cur_loss)
