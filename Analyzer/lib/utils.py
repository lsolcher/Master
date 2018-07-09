import numpy as np
import pickle

def save_obj(obj, name ):
    with open('data/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('data/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def dict_to_list(dictio):
    the_list = []
    for value in dictio.values():
        the_list.append(value)
    return the_list


def lists_to_list_of_sentences(lists):
    one_list = []
    for item in lists:
        one_list += item
    return one_list


def lists_to_list_of_articles(lists):
    one_list = []

    for item in lists:
        one_list.append(''.join(map(str, item)))
    return one_list