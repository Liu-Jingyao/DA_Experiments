from transformers import PretrainedConfig, DistilBertTokenizerFast

from utils import names


class BaseConfig(PretrainedConfig):
    model_type = None
    def __init__(self, vocab_size=40000, num_labels=3, aug_ops=dict(), **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.output_dim = num_labels
        self.aug_ops = aug_ops
