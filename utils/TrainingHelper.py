import datetime
import os
import random
from statistics import mean

import evaluate
import numpy
import numpy as np
import transformers
from transformers import IntervalStrategy
import time
from time import strftime, gmtime
from transformers import set_seed

from utils.DBHelper import save_result
from utils.consts import CUSTOM_MODEL_CONFIG_CLASS_DICT, CUSTOM_MODEL_CLASS_DICT


def compute_metrics(eval_preds):
    metric1 = evaluate.load('accuracy')
    metric2 = evaluate.load('f1')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric2.compute(predictions=predictions, references=labels, average='micro')["f1"]
    return {"f1": f1, "accuracy": accuracy}

class TrainingHelper:
    def __init__(self, task_config, dataset_helper, data_collator, model_config, dataset_config, my_tokenizer, my_logger):
        self.task_config = task_config
        self.dataset_helper = dataset_helper
        self.data_collator = data_collator
        self.big_dataset = dataset_helper.big_dataset
        self.train_size = dataset_helper.train_size
        self.my_logger = my_logger
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.my_tokenizer = my_tokenizer
        self.accs = list()
        self.f1s = list()

    def train_test_loop(self):
        repeat_test_num = self.task_config['repeat_num']
        seeds = [93, 79, 17]
        for i in range(repeat_test_num):
            start_time = time.time()
            seed = seeds[i]
            set_seed(seed)

            self.my_logger.info(f"Test time {i}, seed={seed}")

            # reload model
            if self.model_config['pretrained']:
                checkpoint = self.model_config['checkpoint']
                config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[self.task_config['model']].from_pretrained(checkpoint,
                                                                                                  vocab_size=len(self.my_tokenizer),
                                                                                                  num_labels=self.dataset_config['class_num'],
                                                                                                  aug_ops=self.dataset_helper.current_feature_augmentation_flags,
                                                                                                  seq_len=self.my_tokenizer.max_length)
                model = CUSTOM_MODEL_CLASS_DICT[self.task_config['model']].from_pretrained(checkpoint, config=config_obj, mirror='tuna',
                                                                                                  ignore_mismatched_sizes=True)
                model.tokenizer = self.my_tokenizer
            else:
                config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[self.task_config['model']](vocab_size=len(self.my_tokenizer),
                                                                                  num_labels=self.dataset_config['class_num'],
                                                                                  aug_ops=self.dataset_helper.current_feature_augmentation_flags)
                model = CUSTOM_MODEL_CLASS_DICT[self.task_config['model']](config_obj, self.my_tokenizer)


            train_dataset, validation_dataset, test_dataset = self.dataset_helper.post_split()

            # shuffle dataset
            if self.big_dataset:
                train_dataset = train_dataset.shuffle(buffer_size=self.task_config['map_batch_size'])
            else:
                train_dataset = train_dataset.shuffle(load_from_cache_file=False)

            if self.big_dataset:
                training_params = {'max_steps': self.train_size * self.task_config['epochs'] // self.task_config['batch_size']}
            else:
                training_params = {'num_train_epochs': self.task_config['epochs']}
            train_args = transformers.TrainingArguments(os.path.join("trainer", f"{self.task_config['task_name']}", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                                                             per_device_train_batch_size=self.task_config['batch_size'],
                                                             per_device_eval_batch_size=self.task_config['batch_size'],
                                                             **training_params,
                                                             evaluation_strategy=IntervalStrategy.NO,
                                                             logging_strategy=IntervalStrategy.STEPS,
                                                             logging_steps=self.task_config['logging_steps'],
                                                             report_to=['tensorboard'],
                                                             save_strategy=IntervalStrategy.NO)

            trainer = transformers.Trainer(model, train_args, train_dataset=train_dataset,
                                           eval_dataset=validation_dataset,
                                           data_collator=self.data_collator, compute_metrics=compute_metrics)
            trainer.train()

            res = trainer.evaluate(eval_dataset=test_dataset)
            self.accs.append(res['eval_accuracy'])
            self.f1s.append(res['eval_f1'])
            end_time = time.time()
            run_time = end_time - start_time
            run_time = strftime("%H:%M:%S", gmtime(run_time))
            self.my_logger.info(res)
            save_result(self.task_config, i+1, res['eval_f1'], res['eval_accuracy'], run_time, seed)

        acc = mean(self.accs)
        f1 = mean(self.f1s)

        self.my_logger.info(f"acc={acc}, f1={f1}")