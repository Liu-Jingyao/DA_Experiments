import sys

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

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess
from utils.consts import *
from utils.data_utils import get_custom_tokenizer

task_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
running_config_list = augmentation_config_list = list()
running_config = environment_config = dataset_config = model_config = dict()
# logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('main')
logging.disable(logging.DEBUG)
transformers.utils.logging.set_verbosity_warning()

def init_project():
    # load configuration files
    CONFIG_BASE_PATH = os.path.join(ROOT_PATH, "configs")
    global running_config_list, augmentation_config_list, running_config, environment_config, dataset_config, model_config
    with open(os.path.join(CONFIG_BASE_PATH, "running_configs.yaml"), "r") as f_running_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "model_configs.yaml"), "r") as f_model_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "dataset_configs.yaml"), "r") as f_dataset_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "augmentation_configs.yaml"), "r") as f_augmentation_configs:
        running_config_list = list(yaml.safe_load_all(f_running_configs))
        environment_config = running_config_list[0]
        running_config = running_config_list[1][task_name]
        model_config = yaml.safe_load(f_model_configs)[running_config['model']]
        dataset_config = yaml.safe_load(f_dataset_configs)[running_config['dataset']]
        augmentation_config_list = yaml.safe_load(f_augmentation_configs)
        augmentation_config_list = [aug for aug_name in running_config['augmentations'] for aug in augmentation_config_list
                                     if aug['name'] == aug_name]
    # init system proxy
    if environment_config['environment'] != 'local':
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[environment_config['environment']]

    # init random seeds
    transformers.set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    init_project()
    big_dataset = bool('stream_load' in dataset_config.keys())

    # load, preprocess and tokenize dataset
    dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None), streaming=big_dataset)
    if big_dataset:
        dataset = dataset.shuffle(buffer_size=1_000, seed=SEED)
    else:
        dataset = dataset.shuffle(seed=SEED)
    train_size = dataset_config['splits']['train'] if big_dataset else dataset['train'].num_rows
    eval_size = 2_500 if 'validation' not in dataset.keys() else dataset['validation'].num_rows

    # split train-validation set
    if 'validation' not in dataset.keys():
        if big_dataset:
            train_dataset = dataset['train'].skip(eval_size)
            validation_dataset = dataset['train'].take(eval_size)
            dataset['train'] = train_dataset
            dataset['validation'] = validation_dataset
        else:
            dataset_clean = dataset['train'].train_test_split(train_size=train_size-eval_size, shuffle=False)
            dataset['train'] = dataset_clean.pop('train')
            dataset['validation'] = dataset_clean.pop('test')
        train_size = train_size - eval_size

    # concat text title with content
    if 'title_field' in dataset_config.keys():
        dataset = dataset.map(lambda batch: {dataset_config['text_field']:
                                                 [f"{title}\n{text}"
                                                  for title in batch[dataset_config['title_field']]
                                                  for text in batch[dataset_config['text_field']]]}, batched=True, batch_size=running_config['map_batch_size'])
    # change the label_field name
    if dataset_config['label_field'] != 'label':
        dataset = dataset.rename_column(dataset_config['label_field'], 'label')
    # check whether the label is starting from 0 and the increase rate is 1
    if 'label_dict' in dataset_config.keys():
         dataset = dataset.map(lambda batch: {'label': [dataset_config['label_dict'][ori_label]
                                                        for ori_label in batch['label']]}, batched=True, batch_size=running_config['map_batch_size'])

    text_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'text']
    feature_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'feature']

    # text_space augmentation
    if len(text_augmentations):
        dataset_list = [dataset['train']]
        for augmentation_config in text_augmentations:
            logger.info(f"processing {augmentation_config['name']}")
            data_augmentation = TEXT_DATA_AUGMENTATION_DICT[augmentation_config['name']]
            aug_dataset = dataset['train'].map(lambda batch: data_augmentation(batch, dataset_config['text_field']), batched=True, batch_size=running_config['map_batch_size'])

            if augmentation_config['quality'] == 'low':
                dataset_list.insert(0, aug_dataset)
            else:
                dataset_list.append(aug_dataset)
            logger.info(f"{augmentation_config['name']} down.\n"
                        f"example: origin_text='{aug_dataset[0]['origin_text']}',"
                        f" aug_text='{aug_dataset[0][dataset_config['text_field']]}'")

        dataset['train'] = datasets.concatenate_datasets(dataset_list)
        train_size = dataset['train'] * len(dataset_list)


    # tokenize
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.get('checkpoint', "distilbert-base-uncased"))
    tokenizer = get_custom_tokenizer(tokenizer)
    dataset = dataset.map(lambda batch: tokenizer(batch[dataset_config['text_field']], truncation=True)
                          , batched=True, batch_size=running_config['map_batch_size'], **({'load_from_cache_file': False} if big_dataset or bool(text_augmentations) else {}))
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # split dataset
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    # feature_space augmentation
    for feature_augmentation in feature_augmentations:
        if 'exinfo' in feature_augmentation.keys():
            exinfo = feature_augmentation['exinfo']
            preprocess = CUSTOM_MODEL_PREPROCESS_DICT[exinfo]
            parameters_may_need = {
                'token_field': 'input_ids',
                'tfidf_preprocess': TFIDFPreProcess(train_dataset, vocab_size=len(tokenizer), p=0.025)
                if exinfo == names.DROPOUT_PROB else None,
            }
            train_dataset = train_dataset.map(lambda batch: preprocess(batch, **parameters_may_need), batched=True,
                                  batch_size=running_config['map_batch_size'],
                                  **({'load_from_cache_file': False} if (big_dataset or bool(text_augmentations)) else {}))
            tokenizer.model_input_names += [exinfo]
    cur_feature_augmentation_names = [aug['name'] for aug in feature_augmentations]
    feature_augmentation_flags = {aug_name: aug_name in cur_feature_augmentation_names for aug_name in names.FEATURE_DATA_AUGMENTATIONS}
    logger.info(feature_augmentation_flags)

    # define model, metrics, loss func and train model
    if model_config['pretrained']:
        checkpoint = model_config['checkpoint']
        model_config = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']].from_pretrained(checkpoint, num_labels=dataset_config['class_num'],
                                                                                               aug_ops=feature_augmentation_flags)
        model = CUSTOM_MODEL_CLASS_DICT[running_config['model']].from_pretrained(checkpoint, config=model_config, mirror='tuna')
    else:
        model_config = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']](vocab_size=len(tokenizer), num_labels=dataset_config['class_num'],
                                                                               aug_ops=feature_augmentation_flags)
        model = CUSTOM_MODEL_CLASS_DICT[running_config['model']](model_config)

    def compute_metrics(eval_preds):
        metric = evaluate.load('accuracy')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if big_dataset:
        training_params = {'max_steps': train_size * running_config['epochs'] // running_config['batch_size'],
                           'evaluation_strategy': IntervalStrategy.STEPS,
                           'eval_steps': running_config['eval_steps']}
    else:
        training_params = {'num_train_epochs': running_config['epochs'],
                           'evaluation_strategy': IntervalStrategy.EPOCH}

    train_args = transformers.TrainingArguments("trainer", seed=SEED,
                                                per_device_train_batch_size=running_config['batch_size'],
                                                per_device_eval_batch_size=running_config['batch_size'],
                                                **training_params,
                                                logging_strategy=IntervalStrategy.STEPS, logging_steps=running_config['logging_steps'],
                                                report_to=['tensorboard'], save_strategy=IntervalStrategy.NO,)
    if big_dataset:
        train_dataset = train_dataset.with_format('torch')
        validation_dataset = validation_dataset.with_format('torch')
    trainer = transformers.Trainer(model, train_args, train_dataset=train_dataset,
                                   eval_dataset=validation_dataset,
                                   data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
