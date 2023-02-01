import os
import sys
import yaml
import torch
import random
import datasets
import evaluate
import numpy as np
import transformers
from transformers import IntervalStrategy, BertForSequenceClassification
from utils import *

task_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
dataset_config = model_config = running_config = dict()

def init_project():
    # load configuration files
    CONFIG_BASE_PATH = os.path.join(ROOT_PATH, "configs")
    global dataset_config, model_config, running_config
    with open(os.path.join(CONFIG_BASE_PATH, "running_configs.yaml"), "r") as f_running_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "model_configs.yaml"), "r") as f_model_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "dataset_configs.yaml"), "r") as f_dataset_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "augmentation_configs.yaml"), "r") as f_augmentation_configs:
        running_config = list(yaml.safe_load_all(f_running_configs))
        environment: str = running_config[0]
        running_config = running_config[1][task_name]
        model_config = yaml.safe_load(f_model_configs)[running_config['model']]
        dataset_config = yaml.safe_load(f_dataset_configs)[running_config['dataset']]
        augmentation_config = {k: v for k, v in yaml.safe_load(f_augmentation_configs) if k in running_config['augmentations']}

    # init system proxy
    if environment != 'local':
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[environment]

    # init random seeds
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    init_project()
    checkpoint = model_config['checkpoint']
    big_dataset = bool('stream_load' in dataset_config.keys())

    # load, preprocess and tokenize dataset
    dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None), streaming=big_dataset)
    if big_dataset:
        dataset = dataset.shuffle(buffer_size=1_000, seed=SEED)
    else:
        dataset = dataset.shuffle(seed=SEED)
    train_size = dataset_config['splits']['train'] if big_dataset else dataset['train'].num_rows
    eval_size = 2_500 # if 'validation' not in dataset.keys() else dataset['validation'].num_rows
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
                                                  for text in batch[dataset_config['text_field']]]}, batched=True)
    # change the label_field name
    if dataset_config['label_field'] != 'label':
        dataset = dataset.rename_column(dataset_config['label_field'], 'label')
    # check whether the label is starting from 0 and the growth rate is 1
    if 'label_dict' in dataset_config.keys():
         dataset = dataset.map(lambda batch: {'label': [dataset_config['label_dict'][ori_label]
                                                        for ori_label in batch['label']]}, batched=True)

    # Low-quality text_space augmentation, inserted before the original data

    # High-quality text_space augmentation, appended to the original data

    # tokenize
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    tokenized_dataset = dataset.map(lambda batch:
                                    tokenizer(batch[dataset_config['text_field']], truncation=True), batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # define model, metrics, loss func and train model
    if model_config['pretrained']:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=dataset_config['class_num'])
    else:
        config = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']](num_labels=dataset_config['class_num'])
        model = CUSTOM_MODEL_CLASS_DICT[running_config['model']]()
    def compute_metrics(eval_preds):
        metric = evaluate.load('accuracy')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    train_args = transformers.TrainingArguments("trainer", report_to=['tensorboard'], max_steps=train_size * running_config['epochs'],
                                                save_strategy=IntervalStrategy.STEPS, save_steps=running_config['save_steps'],
                                                evaluation_strategy=IntervalStrategy.STEPS, eval_steps=running_config['eval_steps'],
                                                logging_strategy=IntervalStrategy.STEPS, logging_steps=running_config['logging_steps'],
                                                load_best_model_at_end=True, metric_for_best_model='accuracy', seed=SEED,
                                                disable_tqdm=True, lr_scheduler_type="cosine_with_restarts")
    if big_dataset:
        tokenized_dataset['train'] = tokenized_dataset['train'].with_format('torch')
        tokenized_dataset['validation'] = tokenized_dataset['validation'].with_format('torch')
    trainer = transformers.Trainer(model, train_args, train_dataset=tokenized_dataset['train'],
                                   eval_dataset=tokenized_dataset['validation'],
                                   data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
