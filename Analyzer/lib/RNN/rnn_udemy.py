import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import string
import numpy as np
from collections import Counter


def create_counts(word_list, vocab_size=50000):
    # Begin adding vocab counts with Counter
    vocab = [] + Counter(words).most_common(vocab_size)

    # Turn into a numpy array
    vocab = np.array([word for word, _ in vocab])

    dictionary = {word: code for code, word in enumerate(vocab)}
    data = np.array([dictionary.get(word, 0) for word in words])
    return data, vocab


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def prepare_data(counts, word_list, tokens, articles):
    """

    :param counts: A dictionionary with a Counter for each document
    :param word_list: A dictionary with the wordlist for each document
    :param tokens: A dictionary containing each article where key is the article and token a tuple with [0] = the word and [1] = the lemma
    :param articles:
    :return:
    """
    tf.logging.set_verbosity(tf.logging.DEBUG)
    for key, item in tokens.items():
        if 'JF' in key:
            plain_tokens_JF.append([x[0] for x in item])
        elif 'SPON' in key:
            plain_tokens_SPON.append([x[0] for x in item])
        elif 'ZEIT' in key:
            plain_tokens_ZEIT.append([x[0] for x in item])
    create_counts(word_list)

