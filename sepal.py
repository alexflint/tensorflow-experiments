import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    return tf.data.Dataset.from_tensor_slices((dict(features), labels)).shuffle(1000).repeat().batch(batch_size)


def test_input_fn(features, labels):
    """An input function for validation"""
    return tf.data.Dataset.from_tensor_slices((dict(features), labels)).batch(1)


def create_dataset(train_features, train_labels):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_features), train_labels))
    dataset = dataset.shuffle(1000).repeat().batch(16)
    return dataset


def main():
    batch_size = 16

    train = pd.read_csv("data/iris_training.csv", names=COLUMNS, header=0)
    train_features, train_labels = train, train.pop('Species')

    feature_cols = [tf.feature_column.numeric_column(
        key=col) for col in COLUMNS[:-1]]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir="out/dnn",
    )

    print("training...")
    classifier.train(lambda: train_input_fn(
        train_features, train_labels, batch_size), steps=100)

    # Evaluate the model.
    print("evaluating...")
    eval_result = classifier.evaluate(
        input_fn=lambda: test_input_fn(train_features, train_labels))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    main()
