from lib import Tokenizer, Normalizer, Tagger, Lemmatizer
import time
from collections import Counter
import os
import re
import pickle


def save_obj(obj, name ):
    with open('data/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def has_sub_folder(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.listdir(os.path.join(path_to_parent,fname)):
            return True
    return False


def get_immediate_subdirectories(a_dir):
    return [f.path for f in os.scandir(a_dir) if f.is_dir() ]


def preprocess_tokens_per_document(direct, test_string):
    """
    Takes in a directory, parses the texts in each subdirectory and tokenizes each document
    :param direct:
    :return:
    """
    paths = []
    start = time.time()
    print('tokenizing...')
    if has_sub_folder(direct):
        paths = get_immediate_subdirectories(direct)
    else:
        paths = direct
    tokens = Tokenizer.tokenize_from_dir_to_tokens_per_document(paths)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()

    print('tagging...')
    tokens = Tagger.tag(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('normalizing...')
    tokens = Normalizer.normalize(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('lemmatizing...')
    tokens = Lemmatizer.lemmatize_tokens(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('counting...')
    word_list = {}
    for idx, token in tokens.items():
        word_list[idx] = ([t[0] for t in token])
    counts = {}
    print('counting...')
    for idx, item in word_list.items():
        counts[idx] = Counter(item)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    save_obj(counts, 'counts'+test_string)
    save_obj(word_list, 'word_list'+test_string)
    save_obj(tokens, 'tokens'+test_string)
    return counts, word_list, tokens

def preprocess_tokens_per_source(direct, test_string):
    paths = []
    start = time.time()
    print('tokenizing...')
    if has_sub_folder(direct):
        paths = get_immediate_subdirectories(direct)
    else:
        paths = direct
    tokens = Tokenizer.tokenize_from_dir_to_tokens_per_source(paths)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()

    print('tagging...')
    tokens = Tagger.tag(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('normalizing...')
    tokens = Normalizer.normalize(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('lemmatizing...')
    tokens = Lemmatizer.lemmatize_tokens(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('counting...')
    word_list = {}
    for idx, token in tokens.items():
        word_list[idx] = ([t[0] for t in token])
    counts = {}
    print('counting...')
    for idx, item in word_list.items():
        counts[idx] = Counter(item)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    save_obj(counts, 'counts_source'+test_string)
    save_obj(word_list, 'word_list_source'+test_string)
    save_obj(tokens, 'tokens_source'+test_string)
    return counts, word_list, tokens

def get_articles_from_top_dir(dirpath, test_string):
    start = time.time()
    print('getting text...')
    articles = {}
    if has_sub_folder(dirpath):
        paths = get_immediate_subdirectories(dirpath)
    else:
        paths = dirpath

    if type(paths) is list:
        for path in paths:
            text = ''
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    text = open(os.path.join(path, file), encoding="utf8", errors='ignore').read()
                articles[file] = text
    else:
        for file in os.listdir(paths):
            text = ''
            if file.endswith(".txt"):
                text = open(os.path.join(dirpath, file), encoding="utf8", errors='ignore').read()
                articles[file] = text
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    save_obj(articles, 'articles' + test_string)

    return articles
