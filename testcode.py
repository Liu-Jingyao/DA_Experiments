import datasets
import evaluate
import numpy as np
import torch
import transformers
from torch import Tensor
from transformers import AutoTokenizer, DataCollatorWithPadding, EvalPrediction

from TextCNN import CNNConfig, CNN

dataset = datasets.load_dataset('glue', 'sst2')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

config = CNNConfig(vocab_size=len(tokenizer))
model = CNN(config)


def compute_metrics(eval_preds):
    metric = evaluate.load('accuracy')
    logits, labels = eval_preds
    rounded_preds = torch.round(torch.sigmoid(torch.from_numpy(logits)))
    acc = metric.compute(predictions=rounded_preds, references=labels)
    return {
        'accuracy': acc
    }


train_args = transformers.TrainingArguments("trainer", save_strategy="steps", save_steps=1000,
                                                evaluation_strategy="steps", eval_steps=100,
                                                logging_strategy="steps", logging_steps=100, label_names=['labels'],
                                            disable_tqdm=True)
trainer = transformers.Trainer(model, train_args, train_dataset=tokenized_datasets['train'],
                               data_collator=data_collator,
                               eval_dataset=tokenized_datasets['validation'],
                               compute_metrics=compute_metrics)
trainer.train()