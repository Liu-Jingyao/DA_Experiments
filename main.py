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
from utils.consts import *
from utils.data_utils import get_custom_tokenizer
from evaluate import evaluator

task_name = sys.argv[1] if len(sys.argv) > 1 else 'default'

running_config_list = augmentation_config_list = list()
running_config = environment_config = dataset_config = model_config = text_augmentations = feature_augmentations = dict()
running_name = None
my_logger = None
transformers.utils.logging.set_verbosity_warning()

def init_project():
    # load configuration files
    CONFIG_BASE_PATH = os.path.join(ROOT_PATH, "configs")
    global running_config_list, augmentation_config_list, running_config, environment_config, dataset_config, model_config
    global text_augmentations, feature_augmentations, big_dataset, my_logger, running_name
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
    text_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'text']
    feature_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'feature']

    # init logger
    # logging.root.setLevel(logging.NOTSET)
    model_name = running_config['model']
    dataset_name = running_config['dataset']
    epoch_num = running_config['epochs']
    augs = running_config['augmentations']
    aug_p = running_config['aug_params']
    augs = [f"{aug}_{aug_p[i]}" for i, aug in enumerate(augs)]

    my_logger = logging.getLogger('my_project')
    my_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    running_name = f'{model_name}_{dataset_name}_{epoch_num}epochs_{augs}'
    file_handler = logging.FileHandler(os.path.join(LOG_PATH, running_name + '.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    my_logger.addHandler(file_handler)
    my_logger.addHandler(stdout_handler)


    my_logger.info(f"task_name: {task_name}")

    # init system proxy
    if environment_config['environment'] != names.LOCAL:
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[environment_config['environment']]

    # init random seeds
    # transformers.set_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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
            dataset = dataset.shuffle(buffer_size=running_config['map_batch_size'])
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
                              batched=True, batch_size=running_config['map_batch_size'],
                              load_from_cache_file=load_map_from_cache)
    # change the label_field name
    if dataset_config['label_field'] != 'label':
        dataset = dataset.rename_column(dataset_config['label_field'], 'label')
    # check whether the label is starting from 0 and the increase rate is 1
    if 'label_dict' in dataset_config.keys():
        dataset = dataset.map(lambda batch: {'label': [dataset_config['label_dict'][ori_label]
                                                       for ori_label in batch['label']]}, batched=True,
                              batch_size=running_config['map_batch_size'], load_from_cache_file=load_map_from_cache)
    return dataset

def tokenize(dataset, my_tokenizer):
    dataset = dataset.map(lambda batch: my_tokenizer(batch[dataset_config['text_field']], truncation=True)
                          , batched=True, batch_size=running_config['map_batch_size'],
                          load_from_cache_file=load_map_from_cache)
    max_sequence_length = max(len(x) for x in
                              dataset['train']['input_ids'] + dataset['validation']['input_ids'] + dataset['test'][
                                  'input_ids'])
    my_tokenizer.max_length = max_sequence_length
    return dataset

if __name__ == '__main__':
    init_project()
    big_dataset = bool('stream_load' in dataset_config.keys())

    # load, preprocess and tokenize dataset
    dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None),
                                    streaming=big_dataset, cache_dir=CACHE_DIR)
    train_size = dataset_config['splits']['train'] if big_dataset else dataset['train'].num_rows
    dataset, train_size = split_dataset(dataset, train_size)

    # load map res from cache
    load_map_from_cache = not running_config['ignore_cache']

    # preprocess
    dataset = preprocess(dataset)

    # text_space augmentation
    if len(text_augmentations):
        dataset_list = [dataset['train']]
        for augmentation_config in text_augmentations:
            my_logger.info(f"processing {augmentation_config['name']}")
            data_augmentation = TEXT_DATA_AUGMENTATION_DICT[augmentation_config['name']]
            aug_dataset = dataset['train'].map(lambda batch: data_augmentation(batch, dataset_config['text_field']),
                                               batched=True, batch_size=running_config['map_batch_size'],
                                               load_from_cache_file=load_map_from_cache)

            if augmentation_config['quality'] == 'low':
                dataset_list.insert(0, aug_dataset)
            else:
                dataset_list.append(aug_dataset)
            my_logger.info(f"{augmentation_config['name']} down.\n"
                        f"example: original_text='{aug_dataset[0]['original_text']}',"
                        f" aug_text='{aug_dataset[0][dataset_config['text_field']]}'")

        # dataset['train'] = datasets.concatenate_datasets(dataset_list)
        # train_size = len(dataset['train']) * len(dataset_list)
        dataset['train'] = dataset_list[0]


    # tokenize
    my_tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.get('checkpoint', "distilbert-base-uncased"))
    my_tokenizer = get_custom_tokenizer(my_tokenizer)
    dataset = tokenize(dataset, my_tokenizer)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=my_tokenizer, padding=PaddingStrategy.MAX_LENGTH, max_length=my_tokenizer.max_length)

    # split
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    # feature_space augmentation
    feature_augmentation_flags = {aug_name: None for aug_name in names.FEATURE_DATA_AUGMENTATIONS}
    for i, feature_augmentation in enumerate(feature_augmentations):
        feature_augmentation_flags[feature_augmentation['name']] = running_config['aug_params'][i]

        if 'exinfo' in feature_augmentation.keys():
            exinfo = feature_augmentation['exinfo']
            preprocess = CUSTOM_MODEL_PREPROCESS_DICT[exinfo]
            parameters_may_need = {
                'token_field': 'input_ids',
                'tfidf_preprocess': TFIDFPreProcess(train_dataset, vocab_size=len(my_tokenizer), p=running_config['aug_params'][i])
                if exinfo == names.DROPOUT_PROB else None
            }
            train_dataset = train_dataset.map(lambda batch: preprocess(batch, **parameters_may_need), batched=True,
                                              batch_size=running_config['map_batch_size'], load_from_cache_file=load_map_from_cache)
            my_tokenizer.model_input_names += [exinfo]
    cur_feature_augmentation_names = [aug['name'] for aug in feature_augmentations]
    my_logger.info(feature_augmentation_flags)

    # define model, metrics, loss func and train model
    def compute_metrics(eval_preds):
        metric1 = evaluate.load('accuracy')
        metric2 = evaluate.load('f1')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = metric2.compute(predictions=predictions, references=labels, average='micro')["f1"]
        return {"f1": f1, "accuracy": accuracy}
    if big_dataset:
        training_params = {'max_steps': train_size * running_config['epochs'] // running_config['batch_size']}
    else:
        training_params = {'num_train_epochs': running_config['epochs']}
    train_args = transformers.TrainingArguments(os.path.join("trainer", running_name), #seed=SEED,
                                                per_device_train_batch_size=running_config['batch_size'],
                                                per_device_eval_batch_size=running_config['batch_size'],
                                                **training_params,
                                                evaluation_strategy=IntervalStrategy.STEPS,
                                                eval_steps=running_config['eval_steps'],
                                                logging_strategy=IntervalStrategy.STEPS,
                                                logging_steps=running_config['logging_steps'],
                                                report_to=['tensorboard'],
                                                save_strategy=IntervalStrategy.STEPS,
                                                save_steps=running_config['eval_steps'],
                                                load_best_model_at_end=True,
                                                save_total_limit=5)
    if big_dataset:
        train_dataset = train_dataset.with_format('torch')
        validation_dataset = validation_dataset.with_format('torch')
        test_dataset = test_dataset.with_format('torch')

    accs = list()
    f1s = list()
    # repeated test
    repeat_test_num = running_config['repeat_num']
    for i in range(repeat_test_num):
        my_logger.info(f"Test time {i}")
        # shuffle dataset
        if big_dataset:
            # train_dataset = train_dataset.shuffle(buffer_size=running_config['map_batch_size'], seed=SEED)
            train_dataset = train_dataset.shuffle(buffer_size=running_config['map_batch_size'])
        else:
            # train_dataset = train_dataset.shuffle(seed=SEED)
            train_dataset = train_dataset.shuffle(load_from_cache_file=load_map_from_cache)

        # load model
        if model_config['pretrained']:
            checkpoint = model_config['checkpoint']
            config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']].from_pretrained(checkpoint,
                                                                                                 vocab_size=len(my_tokenizer), num_labels=dataset_config['class_num'],
                                                                                                 aug_ops=feature_augmentation_flags,
                                                                                                 seq_len=my_tokenizer.max_length)
            model = CUSTOM_MODEL_CLASS_DICT[running_config['model']].from_pretrained(checkpoint, config=config_obj, mirror='tuna')
            model.tokenizer = my_tokenizer
        else:
            config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']](vocab_size=len(my_tokenizer), num_labels=dataset_config['class_num'],
                                                                                 aug_ops=feature_augmentation_flags)
            model = CUSTOM_MODEL_CLASS_DICT[running_config['model']](config_obj, my_tokenizer)

        trainer = transformers.Trainer(model, train_args, train_dataset=train_dataset,
                                       eval_dataset=validation_dataset,
                                       data_collator=data_collator, compute_metrics=compute_metrics)
        trainer.train()

        res = trainer.evaluate(eval_dataset=test_dataset)
        accs.append(res['eval_accuracy'])
        f1s.append(res['eval_f1'])
        my_logger.info(res)

    acc = mean(accs)
    f1 = mean(f1s)

    my_logger.info(f"acc={acc}, f1={f1}")
