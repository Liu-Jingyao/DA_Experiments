import copy
import os
import  re
from collections import defaultdict

import nltk
import numpy
import torch
import transformers.tokenization_utils
from nltk.corpus import wordnet

from data_augmentations.EDA import synonym_replacement, get_synonyms
from utils import names
from utils.data_utils import WordClean


def online_replacement(input_ids, replacement_prob=None, attention_mask=None, labels=None, aug_prob=None, model=None, tokenizer=None, aug_name=None, **kwargs):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import gevent
    jobs = [gevent.spawn(replacement_func, input_ids[i], attention_mask[i], replacement_prob[i] if replacement_prob is not None else None, labels[i], aug_prob, model, tokenizer, aug_name) for i in range(len(input_ids))]
    gevent.joinall(jobs)
    new_texts = [job.value for job in jobs]

    tokenized_dict = tokenizer(new_texts, padding=True, truncation=True)
    input_ids = torch.tensor(tokenized_dict['input_ids'], device='cuda:0')
    attention_mask = torch.tensor(tokenized_dict['attention_mask'], device='cuda:0')
    token_type_ids = torch.tensor(tokenized_dict['token_type_ids'], device='cuda:0') if 'token_type_ids' in tokenized_dict.keys() else None

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

def replacement_func(input_ids, attention_mask, replacement_prob, label, aug_prob, model, tokenizer, aug_name, return_dict=False, **kwargs):

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

    # used to display
    output_loss = 0
    tmp_replaced_index = ''
    tmp_new_words = ''
    output_replaced_index = ''
    output_new_words = ''
    replacement_pairs = [list() for _ in range(len(input_ids))]

    model.eval()

    # tf-idf采样
    # mask = torch.logical_and(attention_mask == 1, torch.logical_not(torch.isin(input_ids, torch.Tensor(tokenizer.all_special_ids).cuda())))
    # replacement_prob = torch.where(mask, replacement_prob, torch.zeros_like(replacement_prob))

    sample_index = replacement_prob.multinomial(num_samples=6, replacement=False)
    # sample_index = torch.ones_like(replacement_prob).multinomial(num_samples=6, replacement=False)

    # sorted_index = torch.argsort(replacement_prob, descending=True)
    # sample_index = sorted_index[torch.isin(sorted_index, sample_index)]


    cnt = 0
    for index in sample_index:
        # 遍历Input_ids中下标，解码，标点跳过，尝试替换
        if cnt == 3:
            break
        word = tokenizer.decode(input_ids[index], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not re.match('^[a-zA-Z]+$', word):
                continue
        cnt += 1
        # word_nltk = nltk.word_tokenize(word)
        # pos_tagged = nltk.pos_tag(word_nltk)
        # if pos_tagged[0][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
        if True:
            if aug_name == names.PRED_LOSS_REPLACEMENT:
                max_loss = 0
                max_loss_pred = 0
                max_loss_text = text
            syns = get_synonyms(word)
            if not len(syns):
                cnt -= 1
                continue
            for j, syn in enumerate(syns):
                if j == 5:
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

                replacement_pairs[index].append((index, word, syn, pred, loss))

                if aug_name == names.LOSS_BASED_REPLACEMENT:
                    if loss > max_loss:
                        max_loss = loss
                        best_text = new_text
                elif aug_name == names.PRED_BASED_REPLACEMENT:
                    if pred > max_pred:
                        max_pred = pred
                        best_text = new_text
                elif aug_name == names.PRED_LOSS_REPLACEMENT:
                    if loss + 2*(1-pred)*pred > max_loss + 2*(1-max_loss_pred)*max_loss_pred:
                        max_loss = loss
                        max_loss_pred = pred
                        max_loss_text = new_text
                        tmp_replaced_index = index
                        tmp_new_words = syn
            if aug_name == names.PRED_LOSS_REPLACEMENT:
                if max_loss_pred > max_pred:
                    max_pred = max_loss_pred
                    output_loss = max_loss
                    best_text = max_loss_text
                    output_replaced_index = tmp_replaced_index
                    output_new_words = tmp_new_words

    model.train()

    if return_dict:
        return {'loss': output_loss, 'pred': max_pred, 'text': best_text,
                'replaced_index': output_replaced_index, 'new_words': output_new_words,
                'replacement_pairs': replacement_pairs}

    return best_text