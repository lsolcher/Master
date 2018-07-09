import os
from lib import Tokenizer, Normalizer, text_processor
import numpy as np
from lib.RNN.rnn_classifier_theano import RNNTheano
from datetime import datetime
import sys
import time


_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
_MODEL_FILE = os.environ.get('MODEL_FILE')


def generate_sentence(model, word_to_index, index_to_word):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd_numpy(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            the_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (the_time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.numpy_sdg_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print("Saved model parameters to %s." % outfile)


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    print(nepoch)
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            the_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (the_time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("C:/Programmierung/Masterarbeit/Analyzer/data/trainedModels/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, the_time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
        print("done with ", epoch)


def train_rnn(datapath):
    directory_list = list()
    for root, dirs, files in os.walk(datapath, topdown=False):
        for idx, name in enumerate(dirs):
            if idx == 0:
                directory_list.append(os.path.join(root, name))
    # preprocessing
    sentences = text_processor.get_sentences_from_top_dir(directory_list)
    sentences = Normalizer.set_start_and_end_flags(sentences, sentence_start_token, sentence_end_token)
    tokens = Tokenizer.tokenize_sentences(sentences)
    preprocessed_sentences, word_to_index, index_to_word = Normalizer.remove_uncommon_words_and_index_sentences(tokens, _VOCABULARY_SIZE, unknown_token)

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in preprocessed_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in preprocessed_sentences])
    np.random.seed(10)
    """
    o, s = model.forward_propagation(X_train[10])
    print(o.shape)
    print(o)
    predictions = model.predict(X_train[10])
    # Limit to 1000 examples to save time
    print("Expected Loss for random predictions: %f" % np.log(_VOCABULARY_SIZE))
    print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))
    grad_check_vocab_size = 100
    np.random.seed(10)
    model = RNNClassifier(grad_check_vocab_size, 10, bptt_truncate=1000)
    model.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])
    """

    """
    # Standard Classifier
    model = RNNClassifier(_VOCABULARY_SIZE)
    train_with_sgd(model, X_train[:], y_train[:], nepoch=2, evaluate_loss_after=1)
    num_sentences = 10
    senten_min_length = 7

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model, word_to_index, index_to_word)
        print(" ").join(sent)
    """
    model = RNNTheano(_VOCABULARY_SIZE, hidden_dim=_HIDDEN_DIM)
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
    t2 = time.time()
    print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

    train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)


def rnn(datapath, model):
    directory_list = list()
    for root, dirs, files in os.walk(datapath, topdown=False):
        for idx, name in enumerate(dirs):
            if idx == 0:
                directory_list.append(os.path.join(root, name))
    # preprocessing
    sentences = text_processor.get_sentences_from_top_dir(directory_list)
    sentences = Normalizer.set_start_and_end_flags(sentences, sentence_start_token, sentence_end_token)
    tokens = Tokenizer.tokenize_sentences(sentences)
    preprocessed_sentences, word_to_index, index_to_word = Normalizer.remove_uncommon_words_and_index_sentences(tokens, _VOCABULARY_SIZE, unknown_token)
    num_sentences = 10
    senten_min_length = 7

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model, word_to_index, index_to_word)
        print(" ".join(sent))
    directory_list = list()
    for root, dirs, files in os.walk(datapath, topdown=False):
        for idx, name in enumerate(dirs):
            if idx == 0:
                directory_list.append(os.path.join(root, name))






