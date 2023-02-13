from transformers import PretrainedConfig

from utils import names


class BaseConfig(PretrainedConfig):
    model_type = None
    def __init__(self, vocab_size=40000, num_labels=3, aug_ops=[], **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.output_dim = num_labels
        for aug_name in names.FEATURE_DATA_AUGMENTATIONS:
            setattr(self, aug_name + '_flag', aug_name in aug_ops)