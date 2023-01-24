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
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    init_project()
    checkpoint = model_config["checkpoint"]

    # load, preprocess and tokenize dataset
    dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None))
    # split train-validation set
    if 'validation' not in dataset.keys():
        dataset_clean = dataset["train"].train_test_split(train_size=0.8, seed=42)
        dataset_clean["validation"] = dataset_clean.pop("test")
        dataset_clean["test"] = dataset["test"]
        dataset = dataset_clean
    # concat text title with content
    if 'title_field' in dataset_config.keys():
        dataset.map(lambda example:
                    {dataset_config['text_field']: f"{example[dataset_config['title_field']]}\n"
                                                   f"{example[dataset_config['text_field']]}"})
        dataset.remove_columns(dataset_config['title_field'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    def tokenize_function(examples):
        return tokenizer(examples[dataset_config['text_field']], truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # define model, metrics, loss func and train model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint)
    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "sst2")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    train_args = transformers.TrainingArguments("trainer", evaluation_strategy=IntervalStrategy.EPOCH,
                                                logging_strategy=IntervalStrategy.STEPS,
                                                logging_steps=50, report_to=["tensorboard"])
    trainer = transformers.Trainer(model, train_args, train_dataset=tokenized_dataset['train'],
                                   eval_dataset=tokenized_dataset['validation'], data_collator=data_collator,
                                   compute_metrics=compute_metrics)
    trainer.train()