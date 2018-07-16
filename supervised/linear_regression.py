import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

# for determinism
np.random.seed(0)


def main():
    with tf.summary.FileWriter('out/events') as writer:
        writer.add_graph(tf.get_default_graph())

    input_dims = 2
    output_dims = 1
    num_samples = 500
    batch_size = 20

    g = tf.get_default_graph()

    xs_true = np.random.randn(num_samples, input_dims)
    model_true = np.random.randn(output_dims, input_dims)
    ys_true = np.dot(xs_true, model_true.T)

    # create a dataset
    dataset = tf.data.Dataset.from_tensor_slices({"x": xs_true, "y": ys_true})
    it = dataset.batch(batch_size).repeat().make_one_shot_iterator().get_next()

    x = it["x"]
    y = it["y"]

    # linear model
    model = tf.layers.Dense(units=1, name="model")
    y_predicted = model(x)

    # loss
    loss = tf.losses.mean_squared_error(y, y_predicted)

    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # initialization
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    # run
    s = tf.Session()
    s.run((var_init, table_init))
    for i in range(200):
        try:
            cur_x, cur_y, cur_loss, _ = s.run((x, y, loss, train))
        except tf.errors.OutOfRangeError:
            print("iterator exhausted")
            break

        print("loss:", cur_loss)

    kernel, bias = s.run(model.trainable_weights)

    kernel = np.squeeze(kernel)
    bias = np.squeeze(bias)

    print("estimated bias:")
    print(bias)

    print("estimated kernel:")
    print(kernel)

    print("true kernel:")
    print(model_true)

    print("kernel error:")
    print(np.sum(np.square(kernel - model_true)))

    return


if __name__ == '__main__':
    main()
