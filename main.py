import sys
from statistics import mean

import numpy
import yaml
import torch
import random
import datasets
import evaluate
import numpy as np
import transformers
import logging

from tqdm import tqdm
from transformers import IntervalStrategy
from transformers.utils import PaddingStrategy

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess
from utils.DBHelper import save_result
from utils.DatasetHelper import DatasetHelper
from utils.ProjectConfig import ProjectConfig
from utils.TrainingHelper import TrainingHelper, compute_metrics
from utils.consts import *
from utils.data_utils import get_custom_tokenizer
from evaluate import evaluator

running_name = None
my_logger = None
transformers.utils.logging.set_verbosity_warning()
project_config = ProjectConfig()

def init_project():
    # init system proxy
    if project_config['environment'] != names.LOCAL:
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[project_config['environment']]

def split_dataset(dataset, train_size):
    global dataset_config, running_name
    big_dataset = bool('stream_load' in dataset_config.keys())
    if not big_dataset:
        train_dataset = dataset['train']

        if 'validation' in dataset_config['splits'].keys():
            eval_dataset = dataset['validation']
        else:
            eval_size = 5_000
            temp_dataset = train_dataset.train_test_split(test_size=eval_size)
            train_dataset = temp_dataset.pop('train')
            eval_dataset = temp_dataset.pop('test')
            train_size = train_size - eval_size

        test_dataset = dataset['test']
        dataset = datasets.DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
            'validation': eval_dataset})
    else:
        if 'validation' not in dataset_config['splits'].keys():
            dataset = dataset.shuffle(buffer_size=project_config['map_batch_size'])
            eval_size = 5_000
            train_dataset = dataset['train'].skip(eval_size)
            validation_dataset = dataset['train'].take(eval_size)
            dataset['train'] = train_dataset
            dataset['validation'] = validation_dataset
            train_size = train_size - eval_size
    return dataset, train_size

def preprocess(dataset):
    # concat text title with content
    if 'title_field' in dataset_config.keys():
        dataset = dataset.map(lambda batch: {dataset_config['text_field']:
                                                 [f"{batch[dataset_config['title_field']][i]}\n{text}"
                                                  for i, text in enumerate(batch[dataset_config['text_field']])]},
                              batched=True, batch_size=project_config['map_batch_size'],
                              load_from_cache_file=load_map_from_cache)
    # change the label_field name
    if dataset_config['label_field'] != 'label':
        dataset = dataset.rename_column(dataset_config['label_field'], 'label')
    # check whether the label is starting from 0 and the increase rate is 1
    if 'label_dict' in dataset_config.keys():
        dataset = dataset.map(lambda batch: {'label': [dataset_config['label_dict'][ori_label]
                                                       for ori_label in batch['label']]}, batched=True,
                              batch_size=project_config['map_batch_size'], load_from_cache_file=load_map_from_cache)
    return dataset

def tokenize(dataset, my_tokenizer):
    dataset = dataset.map(lambda batch: my_tokenizer(batch[dataset_config['text_field']], truncation=True)
                          , batched=True, batch_size=project_config['map_batch_size'],
                          load_from_cache_file=load_map_from_cache)
    max_sequence_length = max(len(x) for x in
                              dataset['train']['input_ids'] + dataset['validation']['input_ids'] + dataset['test'][
                                  'input_ids'])
    my_tokenizer.max_length = max_sequence_length
    return dataset

if __name__ == '__main__':
    init_project()

    for running_config, model_config, dataset_config, text_augmentations, feature_augmentations, my_logger, is_new_dataset, is_new_aug in project_config:
        my_logger.info(f"task_name: {running_config['task_name']}")
        load_map_from_cache = not project_config['ignore_cache']
        big_dataset = bool('stream_load' in dataset_config.keys())

        dataset_helper = None
        if is_new_dataset:

            # load, preprocess and tokenize dataset
            dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None),
                                            streaming=big_dataset, cache_dir=CACHE_DIR)
            dataset_helper = DatasetHelper(dataset_config['dataset']+dataset_config.get('subset', None), dataset,
                                           dataset_config, big_dataset, running_config['map_batch_size'])

            dataset_helper.field_regular()
            dataset_helper.split()

        dataset_helper.text_augmentation()

        # tokenize
        my_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config.get('checkpoint', "distilbert-base-uncased"))
        my_tokenizer = get_custom_tokenizer(my_tokenizer)
        data_collator = transformers.DataCollatorWithPadding(tokenizer=my_tokenizer)

        dataset_helper.tokenize(my_tokenizer)

        # feature_space augmentation
        dataset_helper.feature_augmentation(feature_augmentations, my_logger, my_tokenizer)


        # load model
        if model_config['pretrained']:
            checkpoint = model_config['checkpoint']
            config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']].from_pretrained(checkpoint,
                                                                                                 vocab_size=len(
                                                                                                     my_tokenizer),
                                                                                                 num_labels=
                                                                                                 dataset_config[
                                                                                                     'class_num'],
                                                                                                 aug_ops=dataset_helper.current_feature_augmentation_flags,
                                                                                                 seq_len=my_tokenizer.max_length)
            model = CUSTOM_MODEL_CLASS_DICT[running_config['model']].from_pretrained(checkpoint, config=config_obj,
                                                                                     mirror='tuna')
            model.tokenizer = my_tokenizer
        else:
            config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']](vocab_size=len(my_tokenizer),
                                                                                 num_labels=dataset_config['class_num'],
                                                                                 aug_ops=dataset_helper.current_feature_augmentation_flags)
            model = CUSTOM_MODEL_CLASS_DICT[running_config['model']](config_obj, my_tokenizer)

        # train-test
        training_helper = TrainingHelper(running_config, dataset_helper, data_collator, model, my_logger)
        training_helper.train_test_loop()