from lib import Tokenizer, Normalizer, Tagger, Lemmatizer
import time


def preprocess_tokens(direct):
    paths = []
    start = time.time()
    print('tokenizing...')
    for item in direct:
        paths.append(item)
    tokens = Tokenizer.tokenize(paths)
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
    tokens = Lemmatizer.lemmatize(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    return tokens
