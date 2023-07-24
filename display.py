import logging
import os
from typing import Union

import numpy as np
import torch
import transformers
import yaml
from redlines import Redlines

import datasets
from fastapi.middleware.cors import CORSMiddleware


from fastapi import FastAPI

from data_augmentations.EDA import eda, synonym_replacement
from data_augmentations.online_replacement import replacement_func
from data_augmentations.word_dropout import TFIDFPreProcess, tfidf_word_dropout
from utils.DatasetHelper import DatasetHelper
from utils.ProjectConfig import retry
from utils.TrainingHelper import TrainingHelper
from utils.consts import DATASET_CACHE_DIR, CONFIG_BASE_PATH, TOKENIZER_CACHE_DIR, CUSTOM_MODEL_CONFIG_CLASS_DICT, \
    CUSTOM_MODEL_CLASS_DICT, ROOT_PATH
from utils.data_utils import WordClean, get_custom_tokenizer, mark_deleted_words, mark_replaced_word

app = FastAPI()

origins = [
    "http://localhost:*",
    "null",
    None
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/old_augment")
def old_augment(text: str):
    d=eda(text, p_rd=0.2)[0]
    wordclean = WordClean()
    clean_text = wordclean(text)
    words = clean_text.split(' ')
    words = [word for word in words if word != '']
    a_words = synonym_replacement(words, 1)
    r = ' '.join(a_words)

    diff_d = Redlines(clean_text, d)
    diff_r = Redlines(clean_text, r)

    return {"deletion": f"增强产生的新句子: {diff_d.output_markdown}", "replacement": f"增强产生的新句子:{diff_r.output_markdown}"}

@app.get("/my_augment")
def my_augment(text: str, label: int, checkpoint: int):
    with open(os.path.join("configs", "dataset_configs.yaml"), "r") as f_dataset_configs:
        dataset_config_dict = yaml.safe_load(f_dataset_configs)
        dataset_config = dataset_config_dict['rotten_tomatoes']

    # load new tokenizer
    my_tokenizer_path = os.path.join(TOKENIZER_CACHE_DIR, 'distilbert-base-uncased')
    my_tokenizer_config_path = os.path.join(TOKENIZER_CACHE_DIR, 'distilbert-base-uncased', 'config')
    if not os.path.exists(my_tokenizer_path) or not os.path.exists(my_tokenizer_config_path):
        my_tokenizer = retry(transformers.AutoTokenizer.from_pretrained, 'distilbert-base-uncased')
        my_tokenizer_config = retry(transformers.AutoConfig.from_pretrained, 'distilbert-base-uncased')
        my_tokenizer.save_pretrained(my_tokenizer_path)
        my_tokenizer_config.save_pretrained(my_tokenizer_config_path)
    my_tokenizer = transformers.AutoTokenizer.from_pretrained(my_tokenizer_path,
                                                              config=transformers.AutoConfig.from_pretrained(
                                                                  my_tokenizer_config_path))
    my_tokenizer = get_custom_tokenizer(my_tokenizer)

    dataset = datasets.load_dataset('rotten_tomatoes', cache_dir=DATASET_CACHE_DIR)
    dataset_config['train_size'] = 'full_set'
    dataset_helper = DatasetHelper('rotten_tomatoes', dataset, dataset_config, False, 1000)
    dataset_helper.tokenize(my_tokenizer)
    tfidf_preprocess = TFIDFPreProcess(dataset_helper.train_dataset, vocab_size=len(my_tokenizer),
                                                    all_special_ids=my_tokenizer.all_special_ids,
                                                    p=0.2)
    tokenize_result = my_tokenizer(text)
    input_ids = tokenize_result['input_ids']
    attention_mask = tokenize_result['attention_mask']
    dropout_prob = torch.tensor([tfidf_preprocess.get_text_dropout_prob(input_ids)]).cuda()
    replacement_prob = torch.tensor([tfidf_preprocess.get_text_replacement_prob(input_ids)]).cuda()
    input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).cuda()



    if checkpoint == 0:
        config_obj = retry(CUSTOM_MODEL_CONFIG_CLASS_DICT['distilbert'].from_pretrained,
                           'distilbert-base-uncased',
                           vocab_size=len(my_tokenizer),
                           num_labels=dataset_config['class_num'],
                           aug_ops=dataset_helper.current_online_augmentation_flag)
        model = retry(CUSTOM_MODEL_CLASS_DICT['distilbert'].from_pretrained, 'distilbert-base-uncased',
                  config=config_obj, mirror='tuna',
                  ignore_mismatched_sizes=True).cuda()
    else:
        path = os.path.join(ROOT_PATH, 'saved_models', 'distilbert_rotten_tomatoes')
        config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT['distilbert'].from_pretrained(
                           path,
                           vocab_size=len(my_tokenizer),
                           num_labels=dataset_config['class_num'],
                           aug_ops=dataset_helper.current_online_augmentation_flag)
        model = CUSTOM_MODEL_CLASS_DICT['distilbert'].from_pretrained(path,
                  config=config_obj, local_files_only=True,
                  ignore_mismatched_sizes=True).cuda()
    model.tokenizer = my_tokenizer

    dropout_res = tfidf_word_dropout(input_ids, attention_mask, dropout_prob=dropout_prob)
    tfidf = my_tokenizer.decode(dropout_res['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    keep = dropout_res['keep']
    adaptive_res = replacement_func(input_ids[0], attention_mask[0], replacement_prob[0], label, 1, model, tokenizer=my_tokenizer,
                                    aug_name='pred_loss_replacement', return_dict=True)
    adaptive = adaptive_res['text']
    replaced_index = adaptive_res['replaced_index']
    new_words = adaptive_res['new_words']
    loss = adaptive_res['loss']
    pred = adaptive_res['pred']

    input_ids = input_ids.cpu().squeeze().numpy().tolist()
    dropout_prob = dropout_prob.cpu().squeeze().numpy().tolist()
    replacement_prob = replacement_prob.cpu().squeeze().numpy().tolist()
    input_tokens = [my_tokenizer.decode(id) for id in input_ids]
    keep = keep.cpu().squeeze().numpy().tolist()

    output_dropout_probs = [{"word": token, "dropout_prob": "%.3f" % dropout_prob[i], "status": keep[i]} for i, token in enumerate(input_tokens) if token.isalpha()]
    output_replacement_probs = []
    for i, token in enumerate(input_tokens):
        if not token.isalpha():
            continue
        if len(adaptive_res['replacement_pairs'][i]):
            for pair in adaptive_res['replacement_pairs'][i]:
                pred_ = pair[3].item()
                loss_ = pair[4]
                sum = pred_ + loss_
                output_replacement_probs.append({"word": token, "replacement_prob": "%.3f" % replacement_prob[i], "tfidf_status":1,
                                                 "syn": f"{pair[2]} pred={'%.3f' % pair[3].item()} loss={'%.3f' % pair[4]}, w={'%.3f' % sum}",
                                                 "status": 1 if i == replaced_index and pair[2] == new_words else 0})
        else:
            output_replacement_probs.append({"word": token, "replacement_prob": "%.3f" % replacement_prob[i], "tfidf_status":0, "syn": "", "status": 0})

    diff_tfidf = Redlines(text.lower(), tfidf).output_markdown
    diff_adaptive = Redlines(text.lower(), adaptive).output_markdown

    return {"tfidf": f"增强产生的新句子:{diff_tfidf}", "adaptive": f"增强产生的新句子:{diff_adaptive}",
            "dropout_probs": output_dropout_probs, "replacement_probs": output_replacement_probs}


@app.get("/checkpoints")
def get_models():
    return {"status":0, "msg": "", "data":{"options": [{"label": "a", "value": "a"},
     {"label": "b", "value": "b"}]}}

@app.get("/chart")
def get_chart(method: str, dataset: str):
    img_dict = {
        "tfidf": {
          "rotten_tomatoes": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291817768.png",
          "imdb": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291538546.png",
          "yelp-5": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291538547.png",
          "amazon-5": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291538548.png"
        },
        "adaptive": {
            "rotten_tomatoes": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291817040.png",
            "imdb": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291818261.png",
            "yelp-5": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291540594.png",
            "amazon-5": "https://gcore.jsdelivr.net/gh/Liu-Jingyao/imgbed/202304291818724.png"
        }
    }
    return {"status":0, "msg": "", "data":{"img": img_dict[method][dataset]}}