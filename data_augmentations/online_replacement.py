import copy
import os
import  re

import nltk
import numpy
import torch
import transformers.tokenization_utils
from nltk.corpus import wordnet

from data_augmentations.EDA import synonym_replacement, get_synonyms
from utils import names
from utils.data_utils import WordClean


def online_replacement(input_ids, dropout_prob=None, attention_mask=None, labels=None, aug_prob=None, model=None, tokenizer=None, aug_name=None, **kwargs):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import gevent
    jobs = [gevent.spawn(replacement_func, input_ids[i], attention_mask[i], dropout_prob[i], labels[i], aug_prob, model, tokenizer, aug_name) for i in range(len(input_ids))]
    gevent.joinall(jobs)
    new_texts = [job.value for job in jobs]

    tokenized_dict = tokenizer(new_texts, padding=True, truncation=True)
    input_ids = torch.tensor(tokenized_dict['input_ids'], device='cuda:0')
    attention_mask = torch.tensor(tokenized_dict['attention_mask'], device='cuda:0')
    token_type_ids = torch.tensor(tokenized_dict['token_type_ids'], device='cuda:0') if 'token_type_ids' in tokenized_dict.keys() else None

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

def replacement_func(input_ids, attention_mask, dropout_prob, label, aug_prob, model, tokenizer, aug_name, **kwargs):

    # wordclean = WordClean()
    text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if numpy.random.random() > aug_prob:
        return text
    if aug_name == names.ONLINE_RANDOM_REPLACEMENT:
        split_text = nltk.word_tokenize(text)
        return ' '.join(synonym_replacement(split_text, 1))
    # pos_tagged = nltk.pos_tag(split_text)  # [('Just', 'RB'), ('a', 'DT'), ('small', 'JJ'), ('snippet', 'NN'), ('of', 'IN'), ('text', 'NN'), ('.', '.')]

    max_loss = 0
    max_pred = 0
    best_text = text

    model.eval()

    # 取按从小到大排列的元素下标
    mask = attention_mask.nonzero().squeeze().tolist()
    sorted_indices = torch.argsort(dropout_prob).tolist()
    sorted_indices = [i for i in sorted_indices if i in mask]

    # for i, word in enumerate(split_text):
    cnt = 0
    for index in sorted_indices:
        # 遍历Input_ids中下标，解码，遇到dropout=0或标点跳过，尝试替换
        if cnt == 3:
            break
        word = tokenizer.decode(input_ids[index], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not re.match('^[a-zA-Z]+$', word):
                continue
        cnt += 1
        word_nltk = nltk.word_tokenize(word)
        pos_tagged = nltk.pos_tag(word_nltk)
        if pos_tagged[0][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
            if aug_name == names.PRED_LOSS_REPLACEMENT:
                max_loss = 0
                max_loss_pred = 0
                max_loss_text = text
            for j, syn in enumerate(get_synonyms(word)):
                if j == 3:
                    break
                # new_text_split = copy.deepcopy(split_text)
                # new_text_split[i] = syn
                # new_text = " ".join(new_text_split)
                new_input_ids = copy.deepcopy(input_ids).tolist()
                new_syn_ids = tokenizer.encode(syn, add_special_tokens = False)
                new_input_ids[index: index+1] = new_syn_ids
                new_text = tokenizer.decode(new_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                tokenized_dict = tokenizer(new_text, padding=True, truncation=True)
                inner_input_ids = torch.tensor([tokenized_dict['input_ids']], device="cuda:0")
                inner_attention_mask = torch.tensor([tokenized_dict['attention_mask']], device="cuda:0")
                inner_token_type_ids = torch.tensor([tokenized_dict['token_type_ids']], device="cuda:0") if 'token_type_ids' in tokenized_dict.keys() else None

                label = torch.tensor([label], device="cuda:0")

                res_dict = model.forward(input_ids=inner_input_ids, attention_mask=inner_attention_mask, token_type_ids=inner_token_type_ids, labels=label)
                logits, loss = res_dict['logits'], res_dict['loss']
                pred = torch.nn.functional.softmax(logits, dim=1)[:, label]

                if aug_name == names.LOSS_BASED_REPLACEMENT:
                    if loss > max_loss:
                        max_loss = loss
                        best_text = new_text
                elif aug_name == names.PRED_BASED_REPLACEMENT:
                    if pred > max_pred:
                        max_pred = pred
                        best_text = new_text
                elif aug_name == names.PRED_LOSS_REPLACEMENT:
                    if loss > max_loss and pred > max_loss_pred:
                        max_loss = loss
                        max_loss_pred = pred
                        max_loss_text = new_text
            if aug_name == names.PRED_LOSS_REPLACEMENT:
                if max_loss_pred > max_pred:
                    max_pred = max_loss_pred
                    best_text = max_loss_text

    model.train()
    return best_text