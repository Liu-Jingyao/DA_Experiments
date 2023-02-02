import os
from transformers import PreTrainedModel, PretrainedConfig
from typing import Dict
from transformers import DistilBertConfig

from custom_models.ModifierEnhancedBert import ModifierEnhancedBertForSequenceClassification
from custom_models.TextCNN import CNNConfig, CNN
from data_augmentations.Base import BaseAugmentation

# constant vars
SEED = 42
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROXY_DICT = {'vpn': 'http://127.0.0.1:7890', 'quanzhou': 'http://10.55.146.88:12798', 'neimeng': '192.168.1.174:12798'}

CUSTOM_MODEL_CLASS_DICT: Dict[str, type(PreTrainedModel)] = {'cnn': CNN, 'modifier-enhance': ModifierEnhancedBertForSequenceClassification}
CUSTOM_MODEL_CONFIG_CLASS_DICT: Dict[str, type(PretrainedConfig)] = {'cnn': CNNConfig}

DATA_AUGMENTATION_DICT: Dict[str, type(BaseAugmentation)] = {}

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'