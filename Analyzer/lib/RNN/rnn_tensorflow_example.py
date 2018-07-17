import tensorflow as tf
import numpy as np
from .create_word_vector import create_word_vecs
from .rnn_model_tf import RNN
import time
import os
import datetime
import sys

# Data loading params
tf.flags.DEFINE_string("pos_dir", "data/rt-polaritydata/rt-polarity.pos", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "data/rt-polaritydata/rt-polarity.neg", "Path of negative data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train/test data (Default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "vanilla", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def run(articles, articles_test):
    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{} = {}".format(attr.upper(), value))
    print("")
    X_train, X_test, y_train, y_test, word2vec_model = create_word_vecs(articles, articles_test)
    #X_train = np.expand_dims(X_train, axis=2)
    #X_test = np.expand_dims(X_test, axis=2)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)

    #X_train = np.expand_dims(X_train, axis=1)
    #X_test = np.expand_dims(X_test, axis=1)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)
    #text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(7985)

    tf.reset_default_graph() # prevent bugs
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        with sess.as_default():
            rnn =  RNN(
                sequence_length=X_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=7985,
                embedding_size=300,
                cell_type="lstm",
                hidden_size=100,
                l2_reg_lambda=3.0
            )
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(0.01).minimize(rnn.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            #text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Pre-trained word2vec
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = text_vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(rnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = batch_iter(
                list(zip(X_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict_dev = {
                        rnn.input_text: X_test,
                        rnn.input_y: y_test,
                        rnn.dropout_keep_prob: 1.0
                    }
                    summaries_dev, loss, accuracy = sess.run(
                        [dev_summary_op, rnn.loss, rnn.accuracy], feed_dict_dev)
                    dev_summary_writer.add_summary(summaries_dev, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

    """
    num_inputs = 300
    num_neurons = 150
    num_outputs = 3
    learning_rate = 0.0001
    num_train_iterations = 200
    batch_size = 1
    num_time_steps = 30

    vocab_size = 7985
    EMBEDDING_DIM = 300

    # placeholders
    X = tf.placeholder(tf.float32, [None, 300, 1])
    y = tf.placeholder(tf.float32, [None, 3])

    # rnn_model_tf.py cell layer
    cell = tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

    # get output and states of basic rnn cells
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # loss func - MSE
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y)

    # optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()

    # session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    saver = tf.train.Saver()

    # model


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        for it in range(num_train_iterations):
            X_batch, y_batch = next_batch(batch_size, X_train, y_train)
            #print('X_batch shape: ', X_batch.shape)
            #print('y_batch shape: ', y_batch.shape)
            sess.run(train, feed_dict= {X:X_batch, y:y_batch})

            if it % 100 == 0:
                mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
                print(it, "\tMSE", mse)
        y_pred = sess.run(outputs, feed_dict={X: X_test})
        saver.save(sess, './rnn_time_series_model_own')
        print(y_pred)
        print(y_test)

    # eval

    with tf.Session() as sess:
       saver.restore(sess, './rnn_time_series_model_own')
       y_pred = sess.run(outputs, feed_dict={X: X_test})

    print(y_pred)
    print(y_test)
    """