import datetime
import logging
import os
import sys

import pandas as pd
import yaml


from utils.consts import CONFIG_BASE_PATH, LOG_PATH


class ProjectConfig:
    def __init__(self, configs_dir=CONFIG_BASE_PATH):
        self.running_name = sys.argv[1] if len(sys.argv) > 1 else 'default'
        self.task_list = []

        with open(os.path.join(configs_dir, "running_configs.yaml"), "r") as f_running_configs, \
                open(os.path.join(configs_dir, "model_configs.yaml"), "r") as f_model_configs, \
                open(os.path.join(configs_dir, "dataset_configs.yaml"), "r") as f_dataset_configs, \
                open(os.path.join(configs_dir, "augmentation_configs.yaml"), "r") as f_augmentation_configs:
            self.running_config_list = list(yaml.safe_load_all(f_running_configs)) # [global params, single test config, workflow test config]
            self.model_config_dict = yaml.safe_load(f_model_configs)
            self.dataset_config_dict = yaml.safe_load(f_dataset_configs)
            self.augmentation_config_dict = yaml.safe_load(f_augmentation_configs)

        if self.running_name == 'workflow':
            training_config = self.running_config_list[2]['training_config']
            workflow_config = self.running_config_list[2]['workflow_config']

            # generate workflow task list
            if 'baseline' in workflow_config:
                for dataset in workflow_config['baseline']['datasets']:
                    for model in workflow_config['baseline']['models']:
                        task_config = self.running_config_list[0].copy()
                        task_epoch_num = self.running_config_list[2]['training_config'][model][dataset]['epochs']
                        if 'batch_size' in self.running_config_list[2]['training_config'][model][dataset].keys():
                            task_batch_size = self.running_config_list[2]['training_config'][model][dataset]['batch_size']
                        else:
                            task_batch_size = task_config['batch_size']
                        task_name = f"{model}_{dataset}_{task_epoch_num}epochs_baseline"
                        task_config.update({'dataset': dataset, 'model': model,
                                            'epochs': training_config[model][dataset]['epochs'],
                                            'task_name': task_name,
                                            'baseline': True, 'batch_size': task_batch_size,
                                            'augmentations': [], 'aug_params': []})
                        self.task_list.append(task_config)
                workflow_config.pop('baseline')

            new_workflow_config = {}
            for aug, aug_dict in workflow_config.items():
                for dataset in aug_dict['datasets']:
                    for model in aug_dict['models']:
                        if dataset not in new_workflow_config.keys():
                            new_workflow_config[dataset] = dict()
                        if 'augs' not in new_workflow_config[dataset].keys():
                            new_workflow_config[dataset]['augs'] = list()
                        if 'models' not in new_workflow_config[dataset].keys():
                            new_workflow_config[dataset]['models'] = list()
                        new_workflow_config[dataset]['augs'].append((aug, aug_dict['prob']))
                        new_workflow_config[dataset]['models'].append(model)
            for dataset, dataset_dict in new_workflow_config.items():
                for k in dataset_dict.keys():
                  new_workflow_config[dataset][k] = list(dict.fromkeys(dataset_dict[k]))
            # todo 默认为所有的组合，没考虑组合不全的情况
            for dataset, dataset_dict in new_workflow_config.items():
                for aug, aug_prob in dataset_dict['augs']:
                    for model in dataset_dict['models']:
                        task_config = self.running_config_list[0].copy()
                        task_epoch_num = self.running_config_list[2]['training_config'][model][dataset]['epochs']
                        if 'batch_size' in self.running_config_list[2]['training_config'][model][dataset].keys():
                            task_batch_size = self.running_config_list[2]['training_config'][model][dataset]['batch_size']
                        else:
                            task_batch_size = task_config['batch_size']
                        task_name = f"{model}_{dataset}_{task_epoch_num}epochs_[{aug}_{aug_prob}]"
                        task_config.update({'dataset': dataset, 'model': model, 'augmentations': [aug],
                                            'aug_params': [aug_prob], 'epochs': training_config[model][dataset]['epochs'],
                                            'batch_size': task_batch_size,
                                           'task_name': task_name, 'baseline': False})
                        self.task_list.append(task_config)
        else: # single task
            task_config = self.running_config_list[0]
            task_config.update(self.running_config_list[1][self.running_name])
            task_model_name = task_config['model']
            task_dataset_name = task_config['dataset']
            task_epoch_num = task_config['epochs']
            augs = task_config['augmentations']
            aug_p = task_config['aug_params']
            augs = [f"{aug}_{aug_p[i]}" for i, aug in enumerate(augs)]
            task_name = f"{task_model_name}_{task_dataset_name}_{task_epoch_num}epochs_{augs}"
            task_config['task_name'] = task_name
            self.task_list.append(task_config)

        self.current_task_id = 0
        self.current_model_config = None
        self.current_dataset_config = None
        self.current_task_name = None
        self.current_task_config = None
        self.current_offline_augmentations = None
        self.current_online_augmentations = None
        self.current_baseline = None
        self.current_logger = None
        self.prev_dataset = None
        self.new_dataset = None
        self.prev_aug = None
        self.new_aug = None
        self.prev_model = None
        self.new_model = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_task_id < len(self.task_list):
            self.current_task_config = self.task_list[self.current_task_id]
            self.current_task_name = self.current_task_config['task_name']
            self.current_model_config = self.model_config_dict[self.current_task_config['model']]
            self.current_dataset_config = self.dataset_config_dict[self.current_task_config['dataset']]
            augmentation_config_list = [aug for aug_name in self.current_task_config['augmentations']
                                            for aug in self.augmentation_config_dict
                                            if aug['name'] == aug_name]
            for i, aug in enumerate(augmentation_config_list):
                aug['prob'] = self.current_task_config['aug_params'][i]
            self.current_offline_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'offline']
            self.current_online_augmentations = [aug for aug in augmentation_config_list if aug['space'] == 'online']

            self.current_logger = logging.getLogger('my_project')
            self.current_logger.setLevel(logging.INFO)
            self.current_logger.handlers.clear()
            formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)
            file_handler = logging.FileHandler(os.path.join(LOG_PATH,  self.current_task_name + '.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.current_logger.addHandler(file_handler)
            self.current_logger.addHandler(stdout_handler)
            self.new_dataset = self.current_task_config['dataset'] != self.prev_dataset
            self.new_aug = self.current_task_config['augmentations'] and self.current_task_config['augmentations'][0] != self.prev_aug
            self.new_aug = self.new_aug or self.new_dataset
            self.new_model = self.current_task_config['model'] != self.prev_model

            self.current_task_id = self.current_task_id + 1
            self.prev_dataset = self.current_task_config['dataset']
            self.prev_model = self.current_task_config['model']
            self.prev_aug = self.current_task_config['augmentations'][0] if self.current_task_config['augmentations'] else None

            return (self.current_task_config,
                    self.current_model_config,
                    self.current_dataset_config,
                    self.current_offline_augmentations,
                    self.current_online_augmentations,
                    self.current_logger,
                    self.new_dataset,
                    self.new_aug,
                    self.new_model)

        raise StopIteration()

    def __getitem__(self, item):
        return self.running_config_list[0][item]

def retry(func, *args, **kwargs):
    while True:
        try:
            res = func(*args, **kwargs)
        except:
            continue
        break
    return res