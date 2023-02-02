import sys
import yaml
import torch
import random
import datasets
import evaluate
import numpy as np
import transformers
from transformers import IntervalStrategy
from utils import *

task_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
running_config_list = augmentation_config_list = list()
running_config = environment_config = dataset_config = model_config = dict()

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
        augmentation_config_list = [aug for aug in yaml.safe_load(f_augmentation_configs)
                                    if aug['name'] in running_config['augmentations']]
    # init system proxy
    if environment_config['environment'] != 'local':
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[environment_config['environment']]

    # init random seeds
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


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

    # text_space augmentation, inserted before the original data
    text_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'text']
    feature_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'feature']

    for aug_config in text_augmentations:
        data_augmentation = DATA_AUGMENTATION_DICT[aug_config['name']](text_field=dataset_config['text_field'])
        dataset = dataset.map(data_augmentation.embedding, batched=True)
        aug_dataset = dataset.map(data_augmentation.transforms, batched=True)
        if aug_config['quality'] == 'low':
            dataset = datasets.concatenate_datasets([dataset, aug_dataset])
        else:
            dataset = datasets.concatenate_datasets([dataset, aug_dataset])


    # tokenize
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.get('checkpoint', "distilbert-base-uncased"))
    dataset = dataset.map(lambda batch:
                                    tokenizer(batch[dataset_config['text_field']], truncation=True), batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # feature_space augmentation
    if len(feature_augmentations):
        assert len(feature_augmentations) <= 1
        assert feature_augmentations[0]['model'] == running_config['model']
        running_config['model'] = feature_augmentations[0]['name']

    # define model, metrics, loss func and train model
    if len(feature_augmentations) and model_config['pretrained']:
        checkpoint = model_config['checkpoint']
        model = CUSTOM_MODEL_CLASS_DICT[running_config['model']].from_pretrained(checkpoint, num_labels=dataset_config['class_num'], tokenizer=tokenizer)
    elif model_config['pretrained']:
        checkpoint = model_config['checkpoint']
        model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=dataset_config['class_num'])
    else:
        model_config = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']](num_labels=dataset_config['class_num'])
        model = CUSTOM_MODEL_CLASS_DICT[running_config['model']](model_config)

    def compute_metrics(eval_preds):
        metric = evaluate.load('accuracy')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    train_args = transformers.TrainingArguments("trainer", report_to=['tensorboard'], max_steps=train_size * running_config['epochs'] // 8,
                                                save_strategy=IntervalStrategy.STEPS, save_steps=running_config['save_steps'],
                                                evaluation_strategy=IntervalStrategy.STEPS, eval_steps=running_config['eval_steps'],
                                                logging_strategy=IntervalStrategy.STEPS, logging_steps=running_config['logging_steps'],
                                                load_best_model_at_end=True, metric_for_best_model='accuracy', seed=SEED)
    if big_dataset:
        dataset['train'] = dataset['train'].with_format('torch')
        dataset['validation'] = dataset['validation'].with_format('torch')
    trainer = transformers.Trainer(model, train_args, train_dataset=dataset['train'],
                                   eval_dataset=dataset['validation'],
                                   data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
