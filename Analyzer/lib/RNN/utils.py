from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import collections
import os
import sys


def char_to_id(chars):
    return {ch: i for i, ch in enumerate(chars)}


def id_to_chars(chars):
    return {i: ch for i, ch in enumerate(chars)}


def build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()


