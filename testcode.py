import collections
import copy

import datasets
import math
import numpy
import numpy as np

import torch
from transformers import DistilBertForSequenceClassification

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess
from models.DistilBERT import DistilBERTForSequenceClassification

if __name__ == '__main__':
  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
  model