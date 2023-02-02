from abc import ABC
from typing import Dict, List


class BaseAugmentation:
    def __init__(self, text_field):
        self.text_field = text_field

    def transforms(self, batch: Dict[str, List]) -> Dict[str, List]:
        raise NotImplementedError

class BaseFeatureAugmentation(BaseAugmentation, ABC):
    def embedding(self, batch: Dict[str, List]) -> Dict[str, List]:
        raise  NotImplementedError