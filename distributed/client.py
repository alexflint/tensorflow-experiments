import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server")
    args = parser.parse_args()

    c = tf.constant("Hello, distributed TensorFlow!")
    sess = tf.Session(args.server)
    print(sess.run(c))


if __name__ == "__main__":
    main()
