import copy

from data_augmentations.EDA import synonym_replacement, eda
from utils.data_utils import WordClean


def batch_synonym_replacement(batch, text_field, prob):
    aug_batch = []
    wordclean = WordClean()
    for sentence in batch[text_field]:
        sentence = wordclean(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word != '']
        a_words = synonym_replacement(words, max(1, int(prob * len(words))))
        aug_batch.append(' '.join(a_words))

    batch['original_text'] = copy.deepcopy(batch[text_field])
    batch[text_field] = aug_batch
    return batch


def batch_random_deletion(batch, text_field, prob):
    res = [eda(text, p_rd=prob) for text in batch[text_field]]
    batch[text_field] = [_[0] for _ in res]
    batch['original_text'] = [_[-1] for _ in res]
    return batch
