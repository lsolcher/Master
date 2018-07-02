import tensorflow as tf



def train():

    X, y = load_data_and_labels()
    vocab_list, vocab_dict, rev_vocab_dict = create_vocabulary(X)
    X, seq_lens = data_to_token_ids(X, vocab_dict)
    train_X, train_y, train_seq_lens, valid_X, valid_y, valid_seq_lens = \
        split_data(X, y, seq_lens)

    with tf.Session() as sess:

        # Load old model or create new one
        model = create_model(sess, FLAGS)

        # Train results
        for epoch_num, epoch in enumerate(generate_epoch(train_X, train_y,
                train_seq_lens, FLAGS.num_epochs, FLAGS.batch_size)):
            print "EPOCH:", epoch_num

            sess.run(tf.assign(model.lr, FLAGS.learning_rate * \
                (FLAGS.learning_rate_decay_factor ** epoch_num)))

            train_loss = []
            train_accuracy = []
            for batch_num, (batch_X, batch_y, batch_seq_lens) in enumerate(epoch):

                _, loss, accuracy = model.step(sess, batch_X, batch_seq_lens,
                    batch_y, dropout=FLAGS.dropout, forward_only=False,
                    sampling=False)

                train_loss.append(loss)
                train_accuracy.append(accuracy)

            print
            print "EPOCH %i SUMMARY" % epoch_num
            print "Training loss %.3f" % np.mean(train_loss)
            print "Training accuracy %.3f" % np.mean(train_accuracy)
            print "----------------------"

            # Validation results
            for valid_epoch_num, valid_epoch in enumerate(
                generate_epoch(valid_X, valid_y, valid_seq_lens,
                               num_epochs=1, batch_size=FLAGS.batch_size)):
                valid_loss = []
                valid_accuracy = []

                for valid_batch_num, \
                    (valid_batch_X, valid_batch_y, valid_batch_seq_lens) in \
                        enumerate(valid_epoch):

                    loss, accuracy = model.step(sess,
                        valid_batch_X, valid_batch_seq_lens, valid_batch_y,
                        dropout=0.0, forward_only=True, sampling=False)

                    valid_loss.append(loss)
                    valid_accuracy.append(accuracy)

            print "Validation loss %.3f" % np.mean(valid_loss)
            print "Validation accuracy %.3f" % np.mean(valid_accuracy)
            print "----------------------"

            # Save checkpoint every epoch.
            if not os.path.isdir(FLAGS.ckpt_dir):
                os.makedirs(FLAGS.ckpt_dir)
            checkpoint_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
            print "Saving the model."
            model.saver.save(sess, checkpoint_path,
                             global_step=model.global_step)

def sample():

    X, y = load_data_and_labels()
    vocab_list, vocab_dict, rev_vocab_dict = create_vocabulary(X)
    X, seq_lens = data_to_token_ids(X, vocab_dict)

    test_sentence = "It was the worst movie I have ever seen."
    test_sentence = get_tokens(clean_str(test_sentence))
    test_sentence, seq_len = data_to_token_ids([test_sentence], vocab_dict)
    test_sentence = test_sentence[0]
    test_sentence = test_sentence + ([PAD_ID] * (max(len(sentence) \
        for sentence in X) - len(test_sentence)))
    test_sentence = np.array(test_sentence).reshape([1, -1])

    with tf.Session() as sess:
        model = create_model(sess, FLAGS)

        probabilities = model.step(sess, batch_X=test_sentence,
            batch_seq_lens=np.array(seq_len),
            forward_only=True, sampling=True)

        for index, prob in enumerate(probabilities[:seq_len[0]]):
            print rev_vocab_dict[test_sentence[0][index]], prob[1]            return outputs[0], outputs[1], outputs[2]