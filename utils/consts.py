import sys

from transformers import PreTrainedModel, PretrainedConfig
from typing import Dict

from data_augmentations.online_replacement import *
from data_augmentations.word_dropout import *
from models.ALBERT import ALBERTConfig, ALBERTForSequenceClassification
from models.simple_models.CNN import CNNConfig, CNN
from models.DistilBERT import DistilBERTForSequenceClassification, DistilBERTConfig
from models.ELECTRA import ELECTRAForSequenceClassification, ELECTRAConfig
from models.simple_models.LSTM import LSTMConfig, LSTM
from models.simple_models.RNN import RNNConfig, RNN
from models.ROBERTA import ROBERTAForSequenceClassification, ROBERTAConfig
from models.XLNET import XLNETForSequenceClassification, XLNETConfig
from utils import names
from utils.data_utils import TF_IDFExtractor

# environment vars
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ROOT_PATH = os.path.dirname(sys.modules['__main__'].__file__) if hasattr(sys.modules['__main__'], '__file__') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_CACHE_DIR = os.path.join(ROOT_PATH, 'datasets')
TOKENIZER_CACHE_DIR = os.path.join(ROOT_PATH, 'tokenizers')
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
                                                                     names.ROBERTA: ROBERTAConfig,
                                                                     names.ALBERT: ALBERTConfig,
                                                                     names.XLNET: XLNETConfig,
                                                                     names.ELECTRA: ELECTRAConfig}
CUSTOM_MODEL_CLASS_DICT: Dict[str, type(PreTrainedModel)] = {names.CNN: CNN,
                                                             names.DISTILBERT: DistilBERTForSequenceClassification,
                                                             names.LSTM: LSTM,
                                                             names.RNN: RNN,
                                                             names.ROBERTA: ROBERTAForSequenceClassification,
                                                             names.ALBERT: ALBERTForSequenceClassification,
                                                             names.XLNET: XLNETForSequenceClassification,
                                                             names.ELECTRA: ELECTRAForSequenceClassification}
CUSTOM_MODEL_PREPROCESS_DICT: Dict[str, callable] = {names.TFIDFS: TF_IDFExtractor.batch_analyze,
                                                     names.DROPOUT_PROB: TFIDFPreProcess.dropout_prob_batch_preprocess,
                                                     names.REPLACEMENT_PROB: TFIDFPreProcess.replacement_prob_batch_preprocess}

