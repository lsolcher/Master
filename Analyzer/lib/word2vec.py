import gzip
import gensim
import logging
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from lib import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train(counts, word_list, tokens, articles):
    print(tokens)
    plain_tokens_JF = []
    plain_tokens_ZEIT = []
    plain_tokens_SPON = []

    for key, item in tokens.items():
        if 'JF' in key:
            plain_tokens_JF.append([x[0] for x in item])
        elif 'SPON' in key:
            plain_tokens_SPON.append([x[0] for x in item])
        elif 'ZEIT' in key:
            plain_tokens_ZEIT.append([x[0] for x in item])

    w2vmodel_JF = gensim.models.Word2Vec(plain_tokens_JF, iter=10, min_count=10, size=300, workers=4)
    w2vmodel_SPON = gensim.models.Word2Vec(plain_tokens_SPON, iter=10, min_count=10, size=300, workers=4)
    w2vmodel_ZEIT = gensim.models.Word2Vec(plain_tokens_ZEIT, iter=10, min_count=10, size=300, workers=4)

    print(w2vmodel_JF.wv.similarity('FREIHEIT', 'CSU'))
    print(w2vmodel_JF.wv['CSU'])


    # get the most common words
    print(w2vmodel_JF.wv.index2word[0], w2vmodel_JF.wv.index2word[1], w2vmodel_JF.wv.index2word[2])


    # get the least common words
    vocab_size = len(w2vmodel_JF.wv.vocab)
    print(w2vmodel_JF.wv.index2word[vocab_size - 1], w2vmodel_JF.wv.index2word[vocab_size - 2], w2vmodel_JF.wv.index2word[vocab_size - 3])


    #save the models
    utils.save_obj(w2vmodel_JF, 'w2vmodel_JF')
    utils.save_obj(w2vmodel_SPON, 'w2vmodel_SPON')
    utils.save_obj(w2vmodel_ZEIT, 'w2vmodel_ZEIT')
    w2vmodel_JF.save('data/obj/' + 'w2vmodel_JF')
    w2vmodel_SPON.save('data/obj/' + 'w2vmodel_SPON')
    w2vmodel_ZEIT.save('data/obj/' + 'w2vmodel_ZEIT')


    # some similarity fun
    print(w2vmodel_SPON.wv.similarity('SPD', 'CDU'), w2vmodel_SPON.wv.similarity('CDU', 'CSU'))

    # what doesn't fit?
    print(w2vmodel_SPON.wv.doesnt_match("CDU SPD CSU Debatte Zeit".split()))
    print(w2vmodel_JF.wv.doesnt_match("CDU SPD CSU Debatte Zeit".split()))
    print(w2vmodel_ZEIT.wv.doesnt_match("CDU SPD CSU Debatte Zeit".split()))

    #w2v_JF = dict(zip(w2vmodel_JF.wv.index2word, w2vmodel_JF.wv.syn0))
    #w2v_SPON = dict(zip(w2vmodel_SPON.wv.index2word, w2vmodel_SPON.wv.syn0))
    #w2v_ZEIT = dict(zip(w2vmodel_ZEIT.wv.index2word, w2vmodel_ZEIT.wv.syn0))


def load_w2v_model(name):
    model = gensim.models.Word2Vec.load('data/obj/' + name)
    return model
