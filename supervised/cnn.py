import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

# for determinism
np.random.seed(0)

# binary classification
# images will be 10x10
# v1: one class will be all 0s, the other will be all 1s
# v2: circles vs squares, with different offsets

# architecture will be:
#   v1: input -> conv -> dense  (2 output neurons)
#   v2: inputs -> conv -> pool -> conv -> pool -> dense -> dense


def build_classifier(image, num_labels):
    # conv layer
    conv_layer_1 = tf.layers.Conv2D(filters=1, kernel_size=3)
    conv_output_1 = conv_layer_1(image)

    # flatten
    conv_output_flat = tf.reshape(
        conv_output_1, (-1, np.prod(conv_output_1.shape[1:])))

    # dense layer
    dense_layer_1 = tf.layers.Dense(units=num_labels)
    logits = dense_layer_1(conv_output_flat)
    return logits


def main():
    with tf.summary.FileWriter('out/events') as writer:
        writer.add_graph(tf.get_default_graph())

    width = 10
    height = 12
    num_labels = 2
    batch_size = 20

    g = tf.get_default_graph()

    images = np.array((np.zeros((height, width, 1)),
                       np.ones((height, width, 1))))
    labels = np.array((0, 1))

    # create an iterator (not bound to any dataset)
    it = tf.data.Iterator.from_structure(
        output_types={"image": tf.float64, "label": tf.int64},
        output_shapes={"image": tf.TensorShape([None, height, width, 1]), "label": tf.TensorShape([None])})

    # build classifier
    input = it.get_next()
    logits = build_classifier(input["image"], num_labels)

    # loss
    label = input["label"]
    loss = tf.losses.sparse_softmax_cross_entropy(label, logits)

    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # initialization
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    # create a dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        {"image": images, "label": labels})

    # run
    s = tf.Session()
    s.run((var_init, table_init))

    # initialize the iterator to the training set
    training_dataset = dataset.repeat().batch(batch_size)
    s.run(it.make_initializer(training_dataset))
    for _ in range(500):
        cur_loss, _ = s.run((loss, train))
        print("loss:", cur_loss)

    # evaluate
    test_dataset = dataset.batch(1)
    s.run(it.make_initializer(test_dataset))
    for i, label in enumerate(labels):
        out = s.run(logits)
        estimated_label = np.argmax(out)
        print("item {0}:  estimated {1}, labelled {2}  {3}".format(
            i, estimated_label, label, "CORRECT" if label == estimated_label else "INCORRECT"))


if __name__ == '__main__':
    main()
