import re
from ..text_processor import get_sentences_from_sub_dir
from .. import utils
from .. import Tokenizer
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    Adopted to german special chars
    """
    """
    string = re.sub(r"[^A-ZäöüÜÖÄßa-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\\\\", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
    """



def load_data(jf_datapath, zeit_datapath, spon_datapath):
    """
    Loads guardian data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    jf_text = get_sentences_from_sub_dir(jf_datapath)
    zeit_text = get_sentences_from_sub_dir(zeit_datapath)
    spon_text = get_sentences_from_sub_dir(spon_datapath)

    x_labels = []
    for key in jf_text.keys():
        x_labels.append(key)
    for key in zeit_text.keys():
        x_labels.append(key)
    for key in spon_text.keys():
        x_labels.append(key)

    # dict to list
    jf_text = utils.dict_to_list(jf_text)
    zeit_text = utils.dict_to_list(zeit_text)
    spon_text = utils.dict_to_list(spon_text)

    jf_list = utils.lists_to_list_of_articles(jf_text)
    zeit_list = utils.lists_to_list_of_articles(zeit_text)
    spon_list = utils.lists_to_list_of_articles(spon_text)

    # split to words
    jf_string = [s.strip() for s in jf_list]
    zeit_string = [s.strip() for s in zeit_list]
    spon_string = [s.strip() for s in spon_list]

    # Split by words
    x_text = jf_string + zeit_string + spon_string
    x_text = [Tokenizer.tokenize_sentences(sent) for sent in x_text]
    x_strings = []

    for li in x_text:
        stri = ' '.join(li)
        x_strings.append(stri)
    # x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    jf_labels = [[1, 0, 0] for _ in jf_string]
    zeit_labels = [[0, 1, 0] for _ in zeit_string]
    spon_labels = [[0, 0, 1] for _ in spon_string]

    y = np.concatenate([jf_labels, zeit_labels, spon_labels], 0)
    return [x_strings, y, x_labels]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
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
