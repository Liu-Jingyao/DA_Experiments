import random

from data_augmentations.EDA import synonym_replacement
from utils.data_utils import WordClean

if __name__ == '__main__':
  res = []
  for sentence in ['The story prompted so many questions from readers about the symptoms',
                   'treatments and testing for these fungal infections that Saey answered them in a follow-up article online.',
                   'An abridged version appears below.',
                   'People generally get infected by inhaling fungal spores.']:
    if random.random() < 0.5:
      wordclean = WordClean()
      sentence = wordclean(sentence)
      words = sentence.split(' ')
      words = [word for word in words if word != '']
      a_words = synonym_replacement(words, 1)
      res.append(' '.join(a_words))
    else:
      res.append(sentence)
  print(res)