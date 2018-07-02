import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import string
import numpy as np
from collections import Counter

## http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
WORDS_FEATURE = 'words'  # Name of the input words feature.
n_words = 0
EMBEDDING_SIZE = 50
MAX_LABEL = 3

def get_dictionary(word_list):
    count = next(iter(word_list.values()))
    count = count.most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary




def dummy_fun(doc):
    return doc


def train_rnn():
    input_layer = layers.Input((70,))

def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for softmax
  # classification over output classes.
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def get_labels(articles):
    labels = []
    for  key, value in articles.items():
        if 'JF' in key:
            labels.append(0)
        elif 'SPON' in key:
            labels.append(1)
        elif 'ZEIT' in key:
            labels.append(2)
    return labels


def prepare_data(counts, word_list, tokens, articles):
    """

    :param counts: A dictionionary with a Counter for each document
    :param word_list: A dictionary with the wordlist for each document
    :param tokens: A dictionary containing each article where key is the article and token a tuple with [0] = the word and [1] = the lemma
    :param articles:
    :return:
    """
    global n_words
    tf.logging.set_verbosity(tf.logging.DEBUG)

    dbpedia = tf.contrib.learn.datasets.load_dataset(
        'dbpedia', test_with_fake_data=False)
    x_train = pd.Series(dbpedia.train.data[:, 1])
    y_train = pd.Series(dbpedia.train.target)


    trainDF = pd.DataFrame()
    max_document_length = max([len(x.split(" ")) for x in articles.values()])
    trainDF['text'] = articles.values()
    trainDF['label'] = get_labels(articles)
    trainDF['label'] = pd.to_numeric(trainDF['label'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.3)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(10)

    x_transform_train = vocab_processor.fit_transform(X_train)
    x_transform_test = vocab_processor.transform(X_test)

    X_train = np.array(list(x_transform_train))
    X_test = np.array(list(x_transform_test))

    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    model_fn = rnn_model
    classifier = tf.estimator.Estimator(model_fn=model_fn)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: X_train},
        y=y_train,
        batch_size=len(X_train),
        num_epochs=None,
        shuffle=True)

    classifier.train(input_fn=train_input_fn, steps=100)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: X_test}, y=y_test, num_epochs=1, shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

    """
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(10)
    X_train = np.array(list(vocab_processor.fit_transform(X_train)))
    X_test = np.array(list(vocab_processor.transform(X_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    # word tf-ifd
    tfifd_word = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        max_features=5000)
    tfifd_word.fit(trainDF['text'])
    xtrain_tfifd_word = tfifd_word.transform(X_train)
    xtest_tfifd_word = tfifd_word.transform(X_test)

    # ngram tf-ifd
    tfifd_ngram = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        ngram_range=(2, 3),
        max_features = 5000)
    tfifd_ngram.fit(trainDF['text'])
    xtrain_tfifd_ngram = tfifd_ngram.transform(X_train)
    xtest_tfifd_ngram = tfifd_ngram.transform(X_test)
    """
    """
    # char tf-ifd
    tfidf_ngram_chars = TfidfVectorizer(
        analyzer='char',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        ngram_range=(2, 3),
        max_features = 5000)
    tfidf_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars = tfidf_ngram_chars.transform(X_train)
    xtest_tfidf_ngram_chars = tfidf_ngram_chars.transform(X_test)

    trainDF['char_count'] = trainDF['text'].apply(len)
    trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
    trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'] + 1)
    trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    """
    """
    
    def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = tf.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

    vocab_size = len(dictionary)
    n_input = 3
    # number of units in RNN cell
    n_hidden = 512
    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }
    symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
    symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
    """
