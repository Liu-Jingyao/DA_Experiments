from typing import Dict

from data_augmentations import EDA as EDA
from data_augmentations.online_replacement import online_replacement
from data_augmentations.word_dropout import random_word_dropout, tfidf_word_dropout
from utils import names

OFFLINE_DATA_AUGMENTATION_DICT: Dict[str, callable] = {names.SYNONYM_REPLACEMENT: EDA.batch_synonym_replacement,
                                                       names.RANDOM_DELETION: EDA.batch_random_deletion,
                                                       names.RANDOM_SWAP: EDA.batch_random_swap,
                                                       names.RANDOM_INSERTION: EDA.batch_random_insertion}
ONLINE_DATA_AUGMENTATION_DICT: Dict[str, callable] = {names.RANDOM_WORD_DROPOUT: random_word_dropout,
                                                      names.TFIDF_WORD_DROPOUT: tfidf_word_dropout,
                                                      names.ONLINE_RANDOM_REPLACEMENT: online_replacement,
                                                      names.LOSS_BASED_REPLACEMENT: online_replacement,
                                                      names.PRED_BASED_REPLACEMENT: online_replacement,
                                                      names.PRED_LOSS_REPLACEMENT: online_replacement}
