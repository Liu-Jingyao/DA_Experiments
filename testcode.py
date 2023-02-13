import collections
import copy

import datasets
import math
import numpy
import numpy as np

import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess
from models.Base import BaseConfig
from models.DistilBERT import DistilBERTForSequenceClassification

if __name__ == '__main__':
  import logging

  logging.basicConfig(level=logging.NOTSET)
  logger = logging.getLogger('main')
  logging.disable(logging.DEBUG)
  # logging.disable(logging.WARNING)

  logger.info({'a': 1})