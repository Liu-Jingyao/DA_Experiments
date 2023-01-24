import os
import yaml
import torch
import random
import datasets
import evaluate
import numpy as np
import transformers
import pytorch_lightning as pl
from transformers import IntervalStrategy

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL = 'distilbert'  # distilbert, bi-lstm
DATASET = 'sst-2'
RUNNING_WAY = 'train_eval'
RUN_ENVIRONMENT = 'vpn'  # vpn, local, quanzhou
dataset_config = model_config = running_config = dict()


def init_project():
    # load configuration files
    CONFIG_BASE_PATH = os.path.join(ROOT_PATH, "configs")
    global dataset_config, model_config, running_config
    with open(os.path.join(CONFIG_BASE_PATH, "dataset_configs", f'{DATASET}.yaml'), "r") as f1, \
            open(os.path.join(CONFIG_BASE_PATH, "model_configs", f'{MODEL}.yaml'), "r") as f2, \
            open(os.path.join(CONFIG_BASE_PATH, "running_configs", f'{RUNNING_WAY}.yaml'), "r") as f3:
        dataset_config = yaml.safe_load(f1)
        model_config = yaml.safe_load(f2)
        running_config = yaml.safe_load(f3)

    # init system proxy
    PROXY_DICT = {'vpn': 'http://127.0.0.1:7890', 'quanzhou': 'http://10.55.146.88:12798'}
    if RUN_ENVIRONMENT != 'local':
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[RUN_ENVIRONMENT]

    # init random seeds
    SEED = 1234
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    init_project()
    checkpoint = model_config["checkpoint"]

    # process dataset
    dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None))
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # define model, metrics, loss and train
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