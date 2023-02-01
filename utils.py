import os
from TextCNN import CNNConfig, CNN

# constant vars
SEED = 42
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODEL_CLASS_DICT = {'cnn': CNN.__class__}
CUSTOM_MODEL_CONFIG_CLASS_DICT = {'cnn': CNNConfig.__class__}
PROXY_DICT = {'vpn': 'http://127.0.0.1:7890', 'quanzhou': 'http://10.55.146.88:12798', 'neimeng': '192.168.1.174:12798'}