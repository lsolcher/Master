import os
import re


def get_sentences_from_top_dir(dirpath):
    sentences = {}
    for path in dirpath:
        for file in os.listdir(path):
            if file.endswith(".txt"):
                text = open(os.path.join(path, file), encoding="utf8", errors='ignore').read()
                sentences[file] = re.split(r'(?<!\d)\.(?!\d)+', text)  # match any .?! but between numbers and dates
                # TODO: keep delimiters
    return sentences


def get_sentences_from_sub_dir(dirpath):
    sentences = {}
    for file in os.listdir(dirpath):
        if file.endswith(".txt"):
            text = open(os.path.join(dirpath, file), encoding="utf8", errors='ignore').read()
            sentences[file] = re.split(r'(?<!\d)\.(?!\d)+', text)  # match any .?! but between numbers and dates
            # TODO: keep delimiters
    return sentences

