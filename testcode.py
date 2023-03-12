import random

import nltk

from data_augmentations.EDA import synonym_replacement
from utils.data_utils import WordClean

if __name__ == '__main__':
  res = []
  # for sentence in ['The story prompted so many questions from readers about the symptoms',
  #                  'treatments and testing for these fungal infections that Saey answered them in a follow-up article online.',
  #                  'An abridged version appears below.',
  #                  'People generally get infected by inhaling fungal spores.']:
  text_tok = nltk.word_tokenize("People generally get infected by inhaling fungal spores.'")
  print(" ".join(text_tok))