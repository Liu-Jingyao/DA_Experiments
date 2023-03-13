import datasets
import transformers
from utils.DatasetHelper import DatasetHelper
from utils.ProjectConfig import ProjectConfig
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

    for running_config, model_config, dataset_config, text_augmentations, feature_augmentations, my_logger, is_new_dataset, is_new_aug in project_config:
        my_logger.info(f"task_name: {running_config['task_name']}")
        load_map_from_cache = not project_config['ignore_cache']
        big_dataset = bool('stream_load' in dataset_config.keys())

        dataset_helper = None
        if is_new_dataset:

            # load, preprocess and tokenize dataset
            dataset = datasets.load_dataset(dataset_config['dataset'], dataset_config.get('subset', None),
                                            streaming=big_dataset, cache_dir=CACHE_DIR)
            dataset_helper = DatasetHelper(dataset_config['dataset']+dataset_config.get('subset', None), dataset,
                                           dataset_config, big_dataset, running_config['map_batch_size'])

            dataset_helper.field_regular()
            dataset_helper.split()

        # text augmentation
        dataset_helper.text_augmentation()

        # tokenize
        my_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config.get('checkpoint', "distilbert-base-uncased"))
        my_tokenizer = get_custom_tokenizer(my_tokenizer)
        data_collator = transformers.DataCollatorWithPadding(tokenizer=my_tokenizer)

        dataset_helper.tokenize(my_tokenizer)

        # feature_space augmentation
        dataset_helper.feature_augmentation(feature_augmentations, my_logger, my_tokenizer)


        # load model
        if model_config['pretrained']:
            checkpoint = model_config['checkpoint']
            config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']].from_pretrained(checkpoint,
                                                                                                 vocab_size=len(my_tokenizer),
                                                                                                 num_labels=dataset_config['class_num'],
                                                                                                 aug_ops=dataset_helper.current_feature_augmentation_flags,
                                                                                                 seq_len=my_tokenizer.max_length)
            model = CUSTOM_MODEL_CLASS_DICT[running_config['model']].from_pretrained(checkpoint, config=config_obj,
                                                                                     mirror='tuna')
            model.tokenizer = my_tokenizer
        else:
            config_obj = CUSTOM_MODEL_CONFIG_CLASS_DICT[running_config['model']](vocab_size=len(my_tokenizer),
                                                                                 num_labels=dataset_config['class_num'],
                                                                                 aug_ops=dataset_helper.current_feature_augmentation_flags)
            model = CUSTOM_MODEL_CLASS_DICT[running_config['model']](config_obj, my_tokenizer)

        # train-test
        training_helper = TrainingHelper(running_config, dataset_helper, data_collator, model, my_logger)
        training_helper.train_test_loop()