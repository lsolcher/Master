import os
import sys
import tensorflow as tf
import numpy as np
import collections
from .create_word_vector import create_word_vecs


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def run(articles, articles_test):
    X_train, X_test, y_train, y_test = create_word_vecs(articles, articles_test)
    #X_train = np.expand_dims(X_train, axis=2)
    #X_test = np.expand_dims(X_test, axis=2)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)

    tf.reset_default_graph() # prevent bugs
    num_inputs = 300
    num_neurons = 150
    num_outputs = 1
    learning_rate = 0.0001
    num_train_iterations = 2000
    batch_size = 1

    # placeholders
    X = tf.placeholder(tf.float32, [None, 1, num_inputs])
    y = tf.placeholder(tf.float32, [1, 3])

    # RNN cell layer
    cell = tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell,
                                                  output_size=num_outputs)  # wrap the 100 neurons to 1 output cell

    # get output and states of basic rnn cells
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # loss func - MSE
    loss = tf.reduce_mean(tf.square(outputs - y))

    # optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        for it in range(num_train_iterations):
            X_batch, y_batch = next_batch(batch_size, X_train, y_train)
            sess.run(train, feed_dict= {X:X_batch, y:y_batch})

            if it % 100 == 0:
                mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
                print(it, "\tMSE", mse)
        saver.save(sess, './rnn_time_series_model_own')
