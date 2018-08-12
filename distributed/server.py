import tensorflow as tf
server = tf.train.Server.create_local_server()
print(server.target)
server.join()
