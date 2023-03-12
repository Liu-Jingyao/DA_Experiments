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


def pred_loss_replacement(text, label, current_model_pred_loss, inner_attention_mask=None):
    split_text = text.split()
    text = " ".join(split_text)
    text_tok = nltk.word_tokenize(text)
    pos_tagged = nltk.pos_tag(
        text_tok)  # [('Just', 'RB'), ('a', 'DT'), ('small', 'JJ'), ('snippet', 'NN'), ('of', 'IN'), ('text', 'NN'), ('.', '.')]
    # 选取标签保持性最好的句子
    max_pred = 0
    best_text = text
    for i, word in enumerate(split_text):
        # 选取学习代价最高的替换词
        max_loss = 0
        max_loss_pred = 0
        max_loss_text = text
        if pos_tagged[i][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
            for j, syn in enumerate(get_synonyms(word)):
                if j == 5:
                    break
                # print(syn)
                new_text_split = copy.deepcopy(split_text)
                new_text_split[i] = syn
                if inner_attention_mask is None:
                    pred, loss = current_model_pred_loss(text, label)
                else:
                    pred, loss = current_model_pred_loss(text, label, inner_attention_mask)
                if loss > max_loss and pred > max_pred:
                    max_loss = loss
                    max_loss_pred = pred
                    max_loss_text = new_text_split
            if max_loss_pred > max_pred:
                max_pred = max_loss_pred
                best_text = max_loss_text

    # # 选取学习代价最高的句子
    # max_loss = 0
    # best_text = text
    # for i, word in enumerate(split_text):
    #     # 选取情感极性保留最好的替换词
    #     max_pred = 0
    #     max_pred_loss = 0
    #     max_pred_text = text
    #     if pos_tagged[i][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
    #         for j, syn in enumerate(get_synonyms(word)):
    #             if j == 5:
    #                 break
    #             # print(syn)
    #             new_text_split = copy.deepcopy(split_text)
    #             new_text_split[i] = syn
    #             if inner_attention_mask is None:
    #                 pred, loss = current_model_pred_loss(text, label)
    #             else:
    #                 pred, loss = current_model_pred_loss(text, label, inner_attention_mask)
    #             if pred > max_pred:
    #                 max_pred = pred
    #                 max_pred_loss = loss
    #                 max_pred_text = new_text_split
    #         if max_pred_loss > max_loss:
    #             max_loss = max_pred_loss
    #             best_text = max_pred_text

    best_text = ' '.join(best_text)
    return best_text
