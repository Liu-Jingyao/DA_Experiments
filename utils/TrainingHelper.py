import os
from statistics import mean

import evaluate
import numpy as np
import transformers
from transformers import IntervalStrategy

from utils.DBHelper import save_result


def compute_metrics(eval_preds):
    metric1 = evaluate.load('accuracy')
    metric2 = evaluate.load('f1')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric2.compute(predictions=predictions, references=labels, average='micro')["f1"]
    return {"f1": f1, "accuracy": accuracy}

class TrainingHelper:
    def __init__(self, task_config, dataset_helper, data_collator, model, my_logger):
        self.task_config = task_config
        self.dataset_helper = dataset_helper
        self.data_collator = data_collator
        self.big_dataset = dataset_helper.big_dataset
        train_size = dataset_helper.train_size
        if self.big_dataset:
            training_params = {'max_steps': train_size * task_config['epochs'] // task_config['batch_size']}
        else:
            training_params = {'num_train_epochs': task_config['epochs']}
        self.train_args = transformers.TrainingArguments(os.path.join("trainer", task_config['task_name']), #seed=SEED,
                                                    per_device_train_batch_size=task_config['batch_size'],
                                                    per_device_eval_batch_size=task_config['batch_size'],
                                                    **training_params,
                                                    evaluation_strategy=IntervalStrategy.STEPS,
                                                    eval_steps=task_config['eval_steps'],
                                                    logging_strategy=IntervalStrategy.STEPS,
                                                    logging_steps=task_config['logging_steps'],
                                                    report_to=['tensorboard'],
                                                    save_strategy=IntervalStrategy.STEPS,
                                                    save_steps=task_config['eval_steps'],
                                                    # load_best_model_at_end=True, todo
                                                    save_total_limit=5)
        self.my_logger = my_logger
        self.model = model
        self.accs = list()
        self.f1s = list()

    def train_test_loop(self):
        repeat_test_num = self.task_config['repeat_num']
        for i in range(repeat_test_num):
            self.my_logger.info(f"Test time {i}")
            train_dataset, validation_dataset, test_dataset = self.dataset_helper.post_split()

            # shuffle dataset
            if self.big_dataset:
                # train_dataset = train_dataset.shuffle(buffer_size=running_config['map_batch_size'], seed=SEED)
                train_dataset = train_dataset.shuffle(buffer_size=self.task_config['map_batch_size'])
            else:
                # train_dataset = train_dataset.shuffle(seed=SEED)
                train_dataset = train_dataset.shuffle(load_from_cache_file=False)


            trainer = transformers.Trainer(self.model, self.train_args, train_dataset=train_dataset,
                                           eval_dataset=validation_dataset,
                                           data_collator=self.data_collator, compute_metrics=compute_metrics)
            trainer.train()

            res = trainer.evaluate(eval_dataset=test_dataset)
            self.accs.append(res['eval_accuracy'])
            self.f1s.append(res['eval_f1'])
            self.my_logger.info(res)
            save_result(self.task_config, self.task_config.current_baseline, i, res['eval_f1'], res['eval_accuracy'])

        acc = mean(self.accs)
        f1 = mean(self.f1s)

        self.my_logger.info(f"acc={acc}, f1={f1}")