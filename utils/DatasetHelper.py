import copy

import datasets

from data_augmentations.data_augmentation_consts import OFFLINE_DATA_AUGMENTATION_DICT
from data_augmentations.word_dropout import TFIDFPreProcess
from utils import names
from utils.consts import CUSTOM_MODEL_PREPROCESS_DICT


class DatasetHelper:
    def __init__(self, name, dataset, dataset_config, big_dataset=False, map_batch_size=1000):
        self.name = name
        self.dataset_dict = dataset
        self.train_dataset = dataset['train'] = dataset['train'].train_test_split(train_size=500)['train']
        self.eval_dataset = dataset['validation'] if 'validation' in dataset.keys() else None
        self.test_dataset = dataset['test'] if 'test' in dataset.keys() else None
        self.dataset_config = dataset_config
        self.big_dataset = big_dataset
        self.map_batch_size = map_batch_size
        self.current_offline_augmentation_flag = None
        self.current_online_augmentation_flag = {}
        self.train_size = dataset_config['splits']['train'] if big_dataset else dataset['train'].num_rows

        self.split()
        self.field_regular()

        self.original_train_dataset = copy.deepcopy(self.train_dataset)


    def split(self):
        # if 'validation' not in self.dataset_config['splits'].keys():
        #     if not self.big_dataset:
        #         eval_size = 5_000
        #         temp_dataset = self.train_dataset.train_test_split(test_size=eval_size)
        #         self.train_dataset = temp_dataset.pop('train')
        #         self.eval_dataset = temp_dataset.pop('test')
        #
        #         self.dataset_dict = datasets.DatasetDict({
        #             'train': self.train_dataset,
        #             'test': self.test_dataset,
        #             'validation': self.eval_dataset})
        #     else:
        #         self.dataset_dict = self.dataset_dict.shuffle(buffer_size=self.map_batch_size, seed=42)
        #         eval_size = 5_000
        #         train_dataset = self.train_dataset.skip(eval_size)
        #         validation_dataset = self.train_dataset.take(eval_size)
        #         self.dataset_dict['train'] = self.train_dataset = train_dataset
        #         self.dataset_dict['validation'] = self.eval_dataset = validation_dataset
        #     self.train_size = self.train_size - eval_size
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
        # self.eval_dataset = self.dataset_dict['validation']
        self.test_dataset = self.dataset_dict['test']

    def tokenize(self, my_tokenizer):
        self.dataset_dict = self.dataset_dict.remove_columns([column_name for column_name in ['input_ids', 'attention_mask', 'token_type_ids'] if column_name in self.train_dataset.column_names])
        self.dataset_dict = self.dataset_dict.map(lambda batch: my_tokenizer(batch[self.dataset_config['text_field']], truncation=True)
                              , batched=True, batch_size=self.map_batch_size,
                              load_from_cache_file=False)
        self.train_dataset = self.dataset_dict['train']
        # self.eval_dataset = self.dataset_dict['validation']
        self.test_dataset = self.dataset_dict['test']
        # max_sequence_length = max(len(x) for x in
        #                           self.dataset_dict['train']['input_ids'] + self.dataset_dict['validation']['input_ids'] + self.dataset_dict['test']['input_ids'])
        # my_tokenizer.max_length = max_sequence_length

    def offline_augmentation(self, offline_augmentations, my_logger):
        self.current_offline_augmentation_flag = offline_augmentations[0]['name']
        augmentation_config = offline_augmentations[0]
        my_logger.info(f"processing {augmentation_config['name']}")
        data_augmentation = OFFLINE_DATA_AUGMENTATION_DICT[augmentation_config['name']]
        data_augmentation_prob = augmentation_config['prob']
        dataset_to_aug = copy.deepcopy(self.original_train_dataset)
        dataset_to_aug = datasets.concatenate_datasets([dataset_to_aug] * 4)
        aug_dataset = dataset_to_aug.map(lambda batch: data_augmentation(batch, self.dataset_config['text_field'],
                                                                         data_augmentation_prob),
                                                             batched=True, batch_size=self.map_batch_size,
                                                             load_from_cache_file=False)
        my_logger.info(f"{self.current_offline_augmentation_flag} down.\n"
                       f"example: original_text='{aug_dataset[0]['original_text']}',"
                       f" aug_text='{aug_dataset[0][self.dataset_config['text_field']]}'")
        self.dataset_dict['train'] = self.train_dataset = datasets.concatenate_datasets([self.original_train_dataset, aug_dataset])
        self.train_size = self.train_size * 5


    def online_augmentation_preprocess(self, online_augmentations, my_logger, my_tokenizer):
        online_augmentation = online_augmentations[0]
        self.current_online_augmentation_flag = {online_augmentation['name']: online_augmentation['prob']}
        if 'exinfo' in online_augmentation.keys():
            exinfo = online_augmentation['exinfo']
            preprocess = CUSTOM_MODEL_PREPROCESS_DICT[exinfo]
            parameters_may_need = {
                'token_field': 'input_ids',
                'tfidf_preprocess': TFIDFPreProcess(self.train_dataset, vocab_size=len(my_tokenizer),
                                                    all_special_ids=my_tokenizer.all_special_ids,
                                                    p=online_augmentation['prob'])
                if exinfo == names.DROPOUT_PROB else None
            }
            self.dataset_dict['train'] = self.train_dataset = self.train_dataset.map(lambda batch: preprocess(batch, **parameters_may_need), batched=True,
                                              batch_size=self.map_batch_size,
                                              load_from_cache_file=False)
            my_tokenizer.model_input_names += [exinfo]
        # duplicate dataset
        self.dataset_dict['train'] = self.train_dataset = datasets.concatenate_datasets([self.train_dataset] * 5)
        self.train_size = self.train_size * 5
        my_logger.info(f"{online_augmentation['name']} down.")
        my_logger.info(self.current_online_augmentation_flag)

    def post_split(self):
        if self.big_dataset:
            train_dataset = self.train_dataset.with_format('torch')
            # validation_dataset =self.eval_dataset.with_format('torch')
            test_dataset = self.test_dataset.with_format('torch')
        else:
            train_dataset = self.train_dataset
            # validation_dataset =self.eval_dataset
            test_dataset = self.test_dataset
        return train_dataset, None, test_dataset

    def restore(self):
        if self.current_offline_augmentation_flag:
            self.dataset_dict['train'] = self.train_dataset = self.original_train_dataset
            self.current_offline_augmentation_flag = None
            self.train_size = self.train_size // 2
        self.current_online_augmentation_flag = {}


