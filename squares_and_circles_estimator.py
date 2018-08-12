import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy
import tensorflow as tf

# for determinism
np.random.seed(0)

# I took the model in squares_and_circles and embedded it within
# a tensorflow Estimator, in order to:
#  - enable checkpointing
#  - enable tensorboard

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

    return images, labels


def build_cnn(image, num_labels):
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


def build_model(features, labels, mode):
    # feedforward net
    logits = build_cnn(features, 2)
    classes = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")

    # in predict mode we are done at this point
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={
            "classes": classes,
            "probabilities": probabilities
        })
    
    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    # in train mode set up a gradient descent op
    if mode == tf.estimator.ModeKeys.TRAIN:
        # gradient step
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train)

    # finally, in evaluation mode, output an accuracy metric
    accuracy = tf.metrics.accuracy(labels=labels, predictions=classes)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={
        "accuracy": accuracy
    })


def main():
    width = 120
    height = 80
    num_labels = 2
    batch_size = 20
    num_train_examples = 1000
    num_test_examples = 20
    train_steps = 150

    # create classifier
    classifier = tf.estimator.Estimator(
        model_fn=build_model,
        model_dir="log"
    )

    # sample training and test examples
    train_images, train_labels = sample_dataset(num_train_examples, width, height)
    test_images, test_labels = sample_dataset(num_test_examples, width, height)

    # set up dataset
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=np.array(train_images),
        y=np.array(train_labels),
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=np.array(test_images),
        y=np.array(test_labels),
        num_epochs=1,
        shuffle=False)

    # set up logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # train
    classifier.train(input_fn=train_input_fn, steps=train_steps, hooks=[logging_hook])

    # evaluate
    results = classifier.evaluate(eval_input_fn)
    print(results)

if __name__ == '__main__':
    main()
