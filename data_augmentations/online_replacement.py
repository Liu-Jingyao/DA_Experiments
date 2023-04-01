import copy
import os

import nltk
import numpy
import torch
from nltk.corpus import wordnet

from data_augmentations.EDA import synonym_replacement, get_synonyms
from utils import names
from utils.data_utils import WordClean


def online_replacement(input_ids, labels=None, aug_prob=None, model=None, tokenizer=None, aug_name=None, **kwargs):
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import gevent
    jobs = [gevent.spawn(replacement_func, text, labels[i], aug_prob, model, tokenizer, aug_name) for i, text in enumerate(texts)]
    gevent.joinall(jobs)
    new_texts = [job.value for job in jobs]

    tokenized_dict = tokenizer(new_texts, padding=True, truncation=True)
    input_ids = torch.tensor(tokenized_dict['input_ids'], device='cuda:0')
    attention_mask = torch.tensor(tokenized_dict['attention_mask'], device='cuda:0')
    token_type_ids = torch.tensor(tokenized_dict['token_type_ids'], device='cuda:0') if 'token_type_ids' in tokenized_dict.keys() else None

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

def replacement_func(text, label, aug_prob, model, tokenizer, aug_name, **kwargs):
    wordclean = WordClean()
    text = wordclean(text)
    if numpy.random.random() <= aug_prob:
        return text

    split_text = text.split()
    split_text = [word for word in split_text if word != '']
    text = " ".join(split_text)

    if aug_name == names.ONLINE_RANDOM_REPLACEMENT:
        return ' '.join(synonym_replacement(split_text, 1))

    text_tok = nltk.word_tokenize(text)
    pos_tagged = nltk.pos_tag(text_tok)  # [('Just', 'RB'), ('a', 'DT'), ('small', 'JJ'), ('snippet', 'NN'), ('of', 'IN'), ('text', 'NN'), ('.', '.')]
    max_loss = 0
    max_pred = 0
    best_text = text

    model_current_state = copy.deepcopy(model)
    model_current_state.eval()

    for i, word in enumerate(split_text):
        if pos_tagged[i][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'] and word not in wordclean.stoplist:
            if aug_name == names.PRED_LOSS_REPLACEMENT:
                max_loss = 0
                max_loss_pred = 0
                max_loss_text = text
            for j, syn in enumerate(get_synonyms(word)):
                if j == 5:
                    break
                new_text_split = copy.deepcopy(split_text)
                new_text_split[i] = syn
                new_text = " ".join(new_text_split)
                with torch.no_grad():
                    tokenized_dict = tokenizer(new_text, padding=True, truncation=True)
                    input_ids = torch.tensor([tokenized_dict['input_ids']], device="cuda:0")
                    attention_mask = torch.tensor([tokenized_dict['attention_mask']], device="cuda:0")
                    token_type_ids = torch.tensor([tokenized_dict['token_type_ids']], device="cuda:0") if 'token_type_ids' in tokenized_dict.keys() else None

                    label = torch.tensor([label], device="cuda:0")

                    res_dict = model_current_state.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label)
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

    return best_text