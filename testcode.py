import collections
import copy

import datasets
import math
import numpy
import numpy as np

import torch

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess



if __name__ == '__main__':
  bs = 32
  sent_len = 512  # Max len of padded sentences
  V = 4000  # Vocab size
  input = numpy.random.randint(0, V, (bs, sent_len))
  dataset = datasets.Dataset.from_dict({'input_ids': input})
  m = TFIDFPreProcess(dataset['input_ids'], p=0.1)
  dataset = dataset.with_format('torch')
  output = m(dataset['input_ids'])
  print(input)
  print(output)