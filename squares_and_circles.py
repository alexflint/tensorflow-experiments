import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy
import tensorflow as tf

# for determinism
np.random.seed(0)

# binary classification
# images will be:
#   64 x 48
#   circles vs squares, with different offsets
# architecture will be:
#   inputs -> conv -> pool -> conv -> pool -> dense -> dense


def generate_circle(w, h, x, y, r):
    im = np.zeros((h, w))
    for i in range(y-r, y+r+1):
        for j in range(x-r, x+r+1):
            d = np.array((y-i, x-j))
            if np.dot(d, d) < r*r:
                im[i, j] = 1.
    return im


def generate_square(w, h, x, y, r):
    im = np.zeros((h, w))
    im[y-r:y+r+1, x-r:x+r+1] = 1.
    return im


def sample_roi(w, h):
    d = min(w, h)
    r = int(np.random.uniform(d/8, d/4))
    x = int(np.random.uniform(r, w-r-1))
    y = int(np.random.uniform(r, h-r-1))
    return (x, y, r)


def sample_image(w, h, label):
    x, y, r = sample_roi(w, h)
    return generate_square(w, h, x, y, r) if label == 0 else generate_circle(w, h, x, y, r)


def sample_dataset(n, w, h):
    images = []
    labels = []
    for label in range(2):
        for i in range(n):
            image = sample_image(w, h, label)
            labels.append(label)
            images.append(image[:, :, np.newaxis])
            scipy.misc.imsave('out/images_%d_%02d.png' % (label, i), image)

    return images, labels, tf.data.Dataset.from_tensor_slices(
        {"image": np.array(images), "label": np.array(labels)})


def build_classifier(image, num_labels):
    # conv layer #1
    conv1_layer = tf.layers.Conv2D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    conv1 = conv1_layer(image)

    # pool layer #2
    pool1_layer = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2)
    pool1 = pool1_layer(conv1)

    # conv layer #2
    conv2_layer = tf.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    conv2 = conv2_layer(pool1)

    # pool layer #2
    pool2_layer = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2)
    pool2 = pool2_layer(conv2)

    # dense layer #1
    flat = tf.reshape(pool2, (-1, np.prod(pool2.shape[1:])))
    dense1_layer = tf.layers.Dense(
        units=1024,
        activation=tf.nn.relu)
    dense1 = dense1_layer(flat)

    # logit layer
    dense2_layer = tf.layers.Dense(units=num_labels)
    logits = dense2_layer(dense1)

    return logits


def main():
    with tf.summary.FileWriter('out/events') as writer:
        writer.add_graph(tf.get_default_graph())

    width = 120
    height = 80
    num_labels = 2
    batch_size = 20
    num_train_examples = 1000
    num_test_examples = 20
    train_steps = 40

    # get the default tensorflow graph
    g = tf.get_default_graph()

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
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    # initialization
    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    # create training and test datasets
    _, _, train_dataset = sample_dataset(num_train_examples, width, height)
    _, _, test_dataset = sample_dataset(num_test_examples, width, height)

    # run
    s = tf.Session()
    s.run((var_init, table_init))

    # initialize the iterator to the training set
    s.run(it.make_initializer(train_dataset.shuffle(
        buffer_size=num_train_examples).repeat().batch(batch_size)))
    for step in range(train_steps):
        cur_loss, _ = s.run((loss, train))
        print("step {0} of {1}: loss={2:.5f}".format(
            step+1, train_steps, cur_loss))

    # evaluate
    i = 0
    num_correct = 0
    num_incorrect = 0
    s.run(it.make_initializer(test_dataset.batch(1)))
    while True:
        try:
            out, correct_label = s.run((logits, input["label"]))
        except tf.errors.OutOfRangeError:
            print("reached end of sequence")
            break

        estimated_label = np.argmax(out)
        if correct_label == estimated_label:
            num_correct += 1
            flag = "CORRECT"
        else:
            num_incorrect += 1
            flag = "INCORRECT"

        print("item {0}:  estimated {1}, labelled {2}  {3}".format(
            i+1, estimated_label, int(correct_label), flag))

        i += 1

    total = num_correct + num_incorrect

    print("test accuracy: {0}/{1}: {2:.2%}".format(num_correct,
                                                   total,
                                                   float(num_correct)/total))


if __name__ == '__main__':
    main()
