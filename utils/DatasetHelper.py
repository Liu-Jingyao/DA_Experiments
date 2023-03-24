import copy

import datasets

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess
from utils import names
from utils.consts import TEXT_DATA_AUGMENTATION_DICT, CUSTOM_MODEL_PREPROCESS_DICT


class DatasetHelper:
    def __init__(self, name, dataset, dataset_config, big_dataset=False, map_batch_size=1000):
        self.name = name
        self.dataset_dict = dataset
        self.train_dataset = dataset['train']
        self.eval_dataset = dataset['validation'] if 'validation' in dataset.keys() else None
        self.test_dataset = dataset['test'] if 'test' in dataset.keys() else None
        self.dataset_config = dataset_config
        self.big_dataset = big_dataset
        self.map_batch_size = map_batch_size
        self.current_text_augmentation_flag = None
        self.current_feature_augmentation_flags = {aug_name: None for aug_name in names.FEATURE_DATA_AUGMENTATIONS}
        self.train_size = dataset_config['splits']['train'] if big_dataset else dataset['train'].num_rows

        self.split()
        self.field_regular()

        self.original_train_dataset = copy.deepcopy(self.train_dataset)


    def split(self):
        if 'validation' not in self.dataset_config['splits'].keys():
            if not self.big_dataset:
                eval_size = 5_000
                temp_dataset = self.train_dataset.train_test_split(test_size=eval_size)
                self.train_dataset = temp_dataset.pop('train')
                self.eval_dataset = temp_dataset.pop('test')

                self.dataset_dict = datasets.DatasetDict({
                    'train': self.train_dataset,
                    'test': self.test_dataset,
                    'validation': self.eval_dataset})
            else:
                self.dataset_dict = self.dataset_dict.shuffle(buffer_size=self.map_batch_size)
                eval_size = 5_000
                train_dataset = self.train_dataset.skip(eval_size)
                validation_dataset = self.train_dataset.take(eval_size)
                self.dataset_dict['train'] = self.train_dataset = train_dataset
                self.dataset_dict['validation'] = self.eval_dataset = validation_dataset
            self.train_size = self.train_size - eval_size
        if 'test' not in self.dataset_config['splits'].keys():
            self.test_dataset = self.dataset_dict['test'] = copy.deepcopy(self.eval_dataset)

    def field_regular(self):
        # concat text title with content
        if 'title_field' in self.dataset_config.keys():
            self.dataset_dict = self.dataset_dict.map(lambda batch: {self.dataset_config['text_field']:
                                                     [f"{batch[self.dataset_config['title_field']][i]}\n{text}"
                                                      for i, text in enumerate(batch[self.dataset_config['text_field']])]},
                                  batched=True, batch_size=self.map_batch_size,
                                  load_from_cache_file=False)
        # change the label_field name
        if self.dataset_config['label_field'] != 'label':
            self.dataset_dict = self.dataset_dict.rename_column(self.dataset_config['label_field'], 'label')
        # check whether the label is starting from 0 and the increase rate is 1
        if 'label_dict' in self.dataset_config.keys():
            self.dataset_dict = self.dataset_dict.map(lambda batch: {'label': [self.dataset_config['label_dict'][ori_label]
                                                           for ori_label in batch['label']]}, batched=True,
                                  batch_size=self.map_batch_size, load_from_cache_file=False)
        self.train_dataset = self.dataset_dict['train']
        self.eval_dataset = self.dataset_dict['validation']
        self.test_dataset = self.dataset_dict['test']

    def tokenize(self, my_tokenizer):
        self.dataset_dict = self.dataset_dict.map(lambda batch: my_tokenizer(batch[self.dataset_config['text_field']], truncation=True)
                              , batched=True, batch_size=self.map_batch_size,
                              load_from_cache_file=False)
        self.train_dataset = self.dataset_dict['train']
        self.eval_dataset = self.dataset_dict['validation']
        self.test_dataset = self.dataset_dict['test']
        max_sequence_length = max(len(x) for x in
                                  self.dataset_dict['train']['input_ids'] + self.dataset_dict['validation']['input_ids'] + self.dataset_dict['test']['input_ids'])
        my_tokenizer.max_length = max_sequence_length

    def text_augmentation(self, text_augmentations, my_logger):
        self.current_text_augmentation_flag = text_augmentations[0]['name']
        augmentation_config = text_augmentations[0]
        my_logger.info(f"processing {augmentation_config['name']}")
        data_augmentation = TEXT_DATA_AUGMENTATION_DICT[augmentation_config['name']]
        aug_dataset = self.original_train_dataset.map(lambda batch: data_augmentation(batch, self.dataset_config['text_field']),
                                                             batched=True, batch_size=self.map_batch_size,
                                                             load_from_cache_file=False)
        my_logger.info(f"{self.current_text_augmentation_flag} down.\n"
                       f"example: original_text='{aug_dataset[0]['original_text']}',"
                       f" aug_text='{aug_dataset[0][self.dataset_config['text_field']]}'")
        self.dataset_dict['train'] = self.train_dataset = datasets.concatenate_datasets([self.original_train_dataset, aug_dataset])
        self.train_size = self.train_size * 2


    def feature_augmentation(self, feature_augmentations, my_logger, my_tokenizer):
        feature_augmentation = feature_augmentations[0]
        self.current_feature_augmentation_flags = {aug_name: feature_augmentation['prob'] if aug_name ==feature_augmentation['name'] else None
                                                   for aug_name in names.FEATURE_DATA_AUGMENTATIONS}
        if 'exinfo' in feature_augmentation.keys():
            exinfo = feature_augmentation['exinfo']
            preprocess = CUSTOM_MODEL_PREPROCESS_DICT[exinfo]
            parameters_may_need = {
                'token_field': 'input_ids',
                'tfidf_preprocess': TFIDFPreProcess(self.train_dataset, vocab_size=len(my_tokenizer),
                                                    p=feature_augmentation['prob'])
                if exinfo == names.DROPOUT_PROB else None
            }
            self.dataset_dict['train'] = self.train_dataset = self.train_dataset.map(lambda batch: preprocess(batch, **parameters_may_need), batched=True,
                                              batch_size=self.map_batch_size,
                                              load_from_cache_file=False)
            my_tokenizer.model_input_names += [exinfo]
            my_logger.info(f"{feature_augmentation['name']} down.")
        my_logger.info(self.current_feature_augmentation_flags)

    def post_split(self):
        if self.big_dataset:
            train_dataset = self.train_dataset.with_format('torch')
            validation_dataset =self.eval_dataset.with_format('torch')
            test_dataset = self.test_dataset.with_format('torch')
        else:
            train_dataset = self.train_dataset
            validation_dataset =self.eval_dataset
            test_dataset = self.test_dataset
        return train_dataset, validation_dataset, test_dataset

    def restore(self):
        if self.current_text_augmentation_flag:
            self.dataset_dict['train'] = self.train_dataset = self.original_train_dataset
            self.current_text_augmentation_flag = None
            self.train_size = self.train_size // 2