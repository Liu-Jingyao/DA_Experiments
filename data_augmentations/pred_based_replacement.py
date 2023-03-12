import copy
import random

import nltk
from nltk import word_tokenize
from nltk import StanfordTagger
from nltk.corpus import wordnet

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def pred_based_replacement(text, label, current_model_pred, inner_attention_mask=None):
    split_text = text.split()
    text = " ".join(split_text)
    text_tok = nltk.word_tokenize(text)
    pos_tagged = nltk.pos_tag(text_tok)  # [('Just', 'RB'), ('a', 'DT'), ('small', 'JJ'), ('snippet', 'NN'), ('of', 'IN'), ('text', 'NN'), ('.', '.')]
    max_pred = 0
    best_text = text
    for i, word in enumerate(split_text):
        if pos_tagged[i][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
          for j, syn in enumerate(get_synonyms(word)):
            if j == 3:
                break
            # print(syn)
            new_text_split = copy.deepcopy(split_text)
            new_text_split[i] = syn
            if inner_attention_mask is None:
                pred = current_model_pred(text, label)
            else:
                pred = current_model_pred(text, label, inner_attention_mask)
            if pred > max_pred:
              max_pred = pred
              best_text = new_text_split
    best_text = ' '.join(best_text)
    return best_text