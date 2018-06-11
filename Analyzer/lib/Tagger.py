import os
import nltk
import random
import pickle
import ClassifierBasedGermanTagger



def tag(tokens):
    tagger = os.path.join(os.path.dirname(__file__), "nltk_german_classifier_data.pickle")

    with open(tagger, 'rb') as f:
        ger_tagger = pickle.load(f)
    #for t in tokens:
    ger_tagger.tag(['Der', 'kleine', 'gelbe', 'Hund', '.'])

