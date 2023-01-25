import os
import sys
import yaml
import torch
import random
import datasets
import evaluate
import numpy as np
import transformers
from transformers import IntervalStrategy

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
task_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
SEED = 42

dataset_config = model_config = running_config = dict()


def init_project():
    # load configuration files
    CONFIG_BASE_PATH = os.path.join(ROOT_PATH, "configs")
    global dataset_config, model_config, running_config
    with open(os.path.join(CONFIG_BASE_PATH, "running_configs.yaml"), "r") as f_running_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "model_configs.yaml"), "r") as f_model_configs, \
            open(os.path.join(CONFIG_BASE_PATH, "dataset_configs.yaml"), "r") as f_dataset_configs:
        running_config = yaml.safe_load(f_running_configs)
        model_config = yaml.safe_load(f_model_configs)[running_config[task_name]['model']]
        dataset_config = yaml.safe_load(f_dataset_configs)[running_config[task_name]['dataset']]

    # init system proxy
    PROXY_DICT = {'vpn': 'http://127.0.0.1:7890', 'quanzhou': 'http://10.55.146.88:12798'}
    if running_config['environment'] != 'local':
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[running_config['environment']]

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
        dataset = dataset.shuffle(buffer_size=1_000_000, seed=SEED)
    else:
        dataset = dataset.shuffle(seed=SEED)
    train_size = dataset_config['splits']['train'] if big_dataset else dataset['train'].num_rows
    eval_size = 25_000 if 'validation' not in dataset.keys() else dataset['validation'].num_rows

    # split train-validation set
    if 'validation' not in dataset.keys():
        if big_dataset:
            train_dataset = dataset['train'].skip(eval_size)
            validation_dataset = dataset['train'].take(eval_size)
            train_size = train_size - eval_size
            dataset['train'] = train_dataset
            dataset['validation'] = validation_dataset
        else:
            dataset_clean = dataset['train'].train_test_split(train_size=0.8, shuffle=False)
            dataset_clean['validation'] = dataset_clean.pop("test")
            dataset_clean['test'] = dataset['test']
            dataset = dataset_clean
    # concat text title with content
    if 'title_field' in dataset_config.keys():
        dataset = dataset.map(lambda example:
                    {dataset_config['text_field']: f"{example[dataset_config['title_field']]}\n"
                                                   f"{example[dataset_config['text_field']]}"})
    # change the label_field name
    if dataset_config['label_field'] != 'label':
        dataset = dataset.rename_column(dataset_config['label_field'], 'label')
    # check whether the label is starting from 0 and the growth rate is 1
    if 'label_dict' in dataset_config.keys():
        dataset = dataset.map(lambda example: {'label': dataset_config['label_dict'][example['label']]})

    # tokenize
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    def tokenize_function(examples):
        return tokenizer(examples[dataset_config['text_field']], truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # define model, metrics, loss func and train model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=dataset_config['class_num'])
    def compute_metrics(eval_preds):
        metric = evaluate.load('accuracy')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    train_args = transformers.TrainingArguments("trainer", report_to=['tensorboard'], max_steps=train_size * 3,
                                                save_strategy=IntervalStrategy.STEPS, save_steps=1_000, seed=SEED,
                                                evaluation_strategy=IntervalStrategy.STEPS, eval_steps=500,
                                                logging_strategy=IntervalStrategy.STEPS, logging_steps=100,
                                                load_best_model_at_end=True, metric_for_best_model='accuracy')
    trainer = transformers.Trainer(model, train_args, train_dataset=tokenized_dataset['train'].with_format('torch'),
                                   eval_dataset=tokenized_dataset['validation'].with_format('torch'),
                                   data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
