import os
from transformers import PreTrainedModel, PretrainedConfig, DistilBertConfig
from typing import Dict

from data_augmentations.tfidf_word_dropout import TFIDFPreProcess
from models.DistilBERT import DistilBERTForSequenceClassification, DistilBERTConfig
from models.CNN import CNNConfig, CNN
from models.LSTM import LSTMConfig, LSTM
from models.RNN import RNNConfig, RNN
from utils import names
from utils.data_utils import TF_IDFExtractor
import data_augmentations.EDA as EDA

# environment vars
SEED = 42
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ROOT_PATH = os.path.dirname(os.path.abspath("main.py"))
PROXY_DICT = {names.VPN: 'http://127.0.0.1:7890',
              names.QUAN_ZHOU: 'http://10.55.146.88:12798',
              names.NEI_MENG: 'http://192.168.1.174:12798',
              names.BEI_JING: 'http://100.72.64.19:12798'}

# component dicts

CUSTOM_MODEL_CONFIG_CLASS_DICT: Dict[str, type(PretrainedConfig)] = {names.CNN: CNNConfig,
                                                                     names.LSTM: LSTMConfig,
                                                                     names.RNN: RNNConfig,
                                                                     names.DISTILBERT: DistilBERTConfig}
CUSTOM_MODEL_CLASS_DICT: Dict[str, type(PreTrainedModel)] = {names.CNN: CNN,
                                                             names.DISTILBERT: DistilBERTForSequenceClassification,
                                                             names.LSTM: LSTM,
                                                             names.RNN: RNN}
CUSTOM_MODEL_PREPROCESS_DICT: Dict[str, callable] = {names.TFIDF: TF_IDFExtractor.batch_analyze,
                                                     names.DROPOUT_PROB: TFIDFPreProcess.batch_preprocess}

DATA_AUGMENTATION_DICT: Dict[str, callable] = {names.SYNONYM_REPLACEMENT: EDA.batch_synonym_replacement,
                                                             names.RANDOM_DELETION: EDA.batch_random_deletion,
                                                             names.RANDOM_SWAP: EDA.batch_random_swap,
                                                             names.RANDOM_INSERTION: EDA.batch_random_insertion}