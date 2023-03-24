import os
import sys

from transformers import PreTrainedModel, PretrainedConfig, DistilBertConfig, XLMRobertaForSequenceClassification, \
    XLMRobertaConfig, RobertaConfig, RobertaForSequenceClassification, AlbertConfig, AlbertForSequenceClassification, \
    XLNetForSequenceClassification, XLNetConfig, ElectraConfig, T5Config, ElectraForSequenceClassification
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
ROOT_PATH = os.path.dirname(sys.modules['__main__'].__file__)
CACHE_DIR = os.path.join(ROOT_PATH, 'datasets')
LOG_PATH = os.path.join(ROOT_PATH, 'logs')
CONFIG_BASE_PATH = os.path.join(ROOT_PATH, "configs")
PROXY_DICT = {names.VPN: 'http://127.0.0.1:7890',
              names.QUAN_ZHOU: 'http://10.55.146.88:12798',
              names.NEI_MENG: 'http://192.168.1.174:12798',
              names.BEI_JING: 'http://100.72.64.19:12798',
              names.NAN_JING: 'http://172.181.217.43:12798'}

# component dicts

CUSTOM_MODEL_CONFIG_CLASS_DICT: Dict[str, type(PretrainedConfig)] = {names.CNN: CNNConfig,
                                                                     names.LSTM: LSTMConfig,
                                                                     names.RNN: RNNConfig,
                                                                     names.DISTILBERT: DistilBERTConfig,
                                                                     names.ROBERTA: RobertaConfig,
                                                                     names.ALBERT: AlbertConfig,
                                                                     names.XLNET: XLNetConfig,
                                                                     names.ELECTRA: ElectraConfig}
CUSTOM_MODEL_CLASS_DICT: Dict[str, type(PreTrainedModel)] = {names.CNN: CNN,
                                                             names.DISTILBERT: DistilBERTForSequenceClassification,
                                                             names.LSTM: LSTM,
                                                             names.RNN: RNN,
                                                             names.ROBERTA: RobertaForSequenceClassification,
                                                             names.ALBERT: AlbertForSequenceClassification,
                                                             names.XLNET: XLNetForSequenceClassification,
                                                             names.ELECTRA: ElectraForSequenceClassification}
CUSTOM_MODEL_PREPROCESS_DICT: Dict[str, callable] = {names.TFIDFS: TF_IDFExtractor.batch_analyze,
                                                     names.DROPOUT_PROB: TFIDFPreProcess.batch_preprocess}

TEXT_DATA_AUGMENTATION_DICT: Dict[str, callable] = {names.SYNONYM_REPLACEMENT: EDA.batch_synonym_replacement,
                                                    names.RANDOM_DELETION: EDA.batch_random_deletion,
                                                    names.RANDOM_SWAP: EDA.batch_random_swap,
                                                    names.RANDOM_INSERTION: EDA.batch_random_insertion}
