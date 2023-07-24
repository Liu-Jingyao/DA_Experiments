from transformers import DataCollatorForWholeWordMask, AutoConfig

import datasets
import transformers
from utils.DatasetHelper import DatasetHelper
from utils.ProjectConfig import ProjectConfig, retry
from utils.TrainingHelper import TrainingHelper, compute_metrics
from utils.consts import *
from utils.data_utils import get_custom_tokenizer

running_name = None
my_logger = None
transformers.utils.logging.set_verbosity_warning()
project_config = ProjectConfig()

if __name__ == '__main__':
    # init system proxy
    if project_config['environment'] != names.LOCAL:
        os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = PROXY_DICT[project_config['environment']]

    # init tokenizer
    my_tokenizer = retry(transformers.AutoTokenizer.from_pretrained, "distilbert-base-uncased")
    my_tokenizer = get_custom_tokenizer(my_tokenizer)

    # running tasks
    dataset_helper = None
    for task_config, model_config, dataset_config, offline_augmentations, online_augmentations, my_logger, is_new_dataset, is_new_aug, is_new_model in project_config:
        my_logger.info(f"task_name: {task_config['task_name']}")
        load_map_from_cache = not project_config['ignore_cache']
        big_dataset = bool('stream_load' in dataset_config.keys())

        is_new_aug = is_new_dataset = is_new_model = model_config['pretrained'] = True

        if is_new_dataset:
            # load, preprocess and tokenize dataset
            dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None),
                                            streaming=big_dataset, cache_dir=DATASET_CACHE_DIR)
            dataset_helper = DatasetHelper(dataset_config['dataset'] + dataset_config.get('subset', ''), dataset,
                                           dataset_config, big_dataset, task_config['map_batch_size'])

        if is_new_aug:
            # restore augmented dataset
            dataset_helper.restore()

        # offline augmentation
        if offline_augmentations:
            if is_new_aug:
                dataset_helper.offline_augmentation(offline_augmentations, my_tokenizer, my_logger, task_config['n_aug'])

        # tokenize
        if is_new_model and model_config['pretrained']:
            # load new tokenizer
            my_tokenizer_path = os.path.join(TOKENIZER_CACHE_DIR, model_config['checkpoint'])
            my_tokenizer_config_path = os.path.join(TOKENIZER_CACHE_DIR, model_config['checkpoint'], 'config')
            if not os.path.exists(my_tokenizer_path) or not os.path.exists(my_tokenizer_config_path):
                my_tokenizer = retry(transformers.AutoTokenizer.from_pretrained, model_config['checkpoint'])
                my_tokenizer_config = retry(transformers.AutoConfig.from_pretrained, model_config['checkpoint'])
                my_tokenizer.save_pretrained(my_tokenizer_path)
                my_tokenizer_config.save_pretrained(my_tokenizer_config_path)
            my_tokenizer = transformers.AutoTokenizer.from_pretrained(my_tokenizer_path, config=transformers.AutoConfig.from_pretrained(my_tokenizer_config_path))
            my_tokenizer = get_custom_tokenizer(my_tokenizer)
        # todo non-pretrained model should use default tokenizer
        data_collator = transformers.DataCollatorWithPadding(tokenizer=my_tokenizer)
        if is_new_dataset or is_new_aug or (is_new_model and model_config['pretrained']):
            dataset_helper.tokenize(my_tokenizer)

        # online augmentation preprocess
        if (is_new_aug or (is_new_model and model_config['pretrained'])) and online_augmentations:
            dataset_helper.online_augmentation_preprocess(online_augmentations, my_logger, my_tokenizer, task_config['n_aug'])

        # train-test
        training_helper = TrainingHelper(task_config, dataset_helper, data_collator, model_config, dataset_config, my_tokenizer, my_logger)
        training_helper.train_test_loop()
