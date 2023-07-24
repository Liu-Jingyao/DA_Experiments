import os
import re
from abc import ABC
import difflib

import torch
from gensim import corpora
from gensim import  models
import itertools

import copy
import json
import os
import re
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from packaging import version

from torch import TensorType
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.tokenization_utils_base import EncodedInput
from transformers.utils import PaddingStrategy, is_tf_tensor, is_torch_tensor, to_py_obj

from utils.names import EX_FEATURE_NAMES


def get_custom_tokenizer(tokenizer: PreTrainedTokenizerBase):
    class CustomTokenizer(tokenizer.__class__, ABC):
        def _pad(self, encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding], max_length: Optional[int] = None,
                 padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD, pad_to_multiple_of: Optional[int] = None,
                 return_attention_mask: Optional[bool] = None) -> dict:

            if return_attention_mask is None:
                return_attention_mask = "attention_mask" in self.model_input_names

            required_input = encoded_inputs[self.model_input_names[0]]

            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)

            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD

            # Initialize attention mask if not present.
            if return_attention_mask and "attention_mask" not in encoded_inputs:
                encoded_inputs["attention_mask"] = [1] * len(required_input)

            if needs_to_be_padded:
                difference = max_length - len(required_input)

                if self.padding_side == "right":
                    if return_attention_mask:
                        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    if "token_type_ids" in encoded_inputs:
                        token_type_ids_difference = max_length - len(encoded_inputs["token_type_ids"])
                        encoded_inputs["token_type_ids"] = (
                            encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * token_type_ids_difference
                        )
                    if "special_tokens_mask" in encoded_inputs:
                        encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                    for custom_input in EX_FEATURE_NAMES:
                        if custom_input in encoded_inputs:
                            custom_difference = max_length - len(encoded_inputs[custom_input])
                            encoded_inputs[custom_input] = encoded_inputs[custom_input] + [0] * custom_difference
                    encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                elif self.padding_side == "left":
                    if return_attention_mask:
                        encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                    if "token_type_ids" in encoded_inputs:
                        token_type_ids_difference = max_length - len(encoded_inputs["token_type_ids"])
                        encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * token_type_ids_difference + encoded_inputs[
                            "token_type_ids"
                        ]
                    if "special_tokens_mask" in encoded_inputs:
                        encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                    for custom_input in EX_FEATURE_NAMES:
                        if custom_input in encoded_inputs:
                            custom_difference = max_length - len(encoded_inputs[custom_input])
                            encoded_inputs[custom_input] = [0] * custom_difference + encoded_inputs[custom_input]
                    encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

            return encoded_inputs

    tokenizer.__class__ = CustomTokenizer
    return tokenizer

class TF_IDFExtractor:
    @staticmethod
    def get_token_tfidf_list(token_id_lists):
        texts = [[str(word) for word in document] for document in token_id_lists]
        processed_corpus = texts
        dictionary = corpora.Dictionary(processed_corpus)
        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        tfidf = models.TfidfModel(bow_corpus)
        word_tfidf_list = []
        id2token = {item[1]: item[0] for item in dictionary.token2id.items()}
        for text in processed_corpus:
            word_tfidf = {id2token[_[0]]: _[1] for _ in tfidf[dictionary.doc2bow(text)]}
            word_tfidf_list.append([word_tfidf.get(token, 0) for token in text])
        return word_tfidf_list

    @staticmethod
    def get_keyword_list(token_id_lists, tfidf_list):
        keyword_list = []
        for instance_num, instance_tfidf in enumerate(tfidf_list):
            keyword_index = instance_tfidf.index(max(instance_tfidf))
            keyword_list.append([token_id_lists[instance_num][keyword_index]])
        return keyword_list

    @staticmethod
    def batch_analyze(batch, token_field, **kwargs):
        tfidf_list =  TF_IDFExtractor.get_token_tfidf_list(batch[token_field])
        batch['tfidfs'] = tfidf_list
        keyword_list = TF_IDFExtractor.get_keyword_list(batch[token_field], tfidf_list)
        batch['keyword_ids'] = keyword_list
        return batch

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class WordClean:
    def __init__(self):
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        self.stoplist = set(stopwords.words('english'))
        self.stoplist.update(('cm', 'kg', 'mr', 'wa', 'nv', 'ore', 'da', 'pm', 'am', 'cx'))
        self.stoplist.remove('not')
        self.APPOS_LOWERCASE = {"ain't": "am not", "aren't": "are not", "can't": "cannot",
                 "can't've": "cannot have", "'cause": "because",
                 "could've": "could have", "couldn't": "could not",
                 "couldn't've": "could not have", "didn't": "did not",
                 "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                 "hadn't've": "had not have", "hasn't": "has not",
                 "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
                 "he'll": "he will", "he'll've": "he will have",
                 "he's": "he is", "how'd": "how did",
                 "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                 "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                                "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                                "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                                "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                                "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                                "might've": "might have", "mightn't": "might not",
                                "mightn't've": "might not have", "must've": "must have",
                                "mustn't": "must not", "mustn't've": "must not have",
                                "needn't": "need not", "needn't've": "need not have",
                                "o'clock": "of the clock", "oughtn't": "ought not",
                                "oughtn't've": "ought not have", "shan't": "shall not",
                                "sha'n't": "shall not", "shan't've": "shall not have",
                                "she'd": "she would", "she'd've": "she would have",
                                "she'll": "she will", "she'll've": "she will have",
                                "she's": "she is", "should've": "should have",
                                "shouldn't": "should not", "shouldn't've": "should not have",
                                "so've": "so have", "so's": "so is",
                                "that'd": "that had", "that'd've": "that would have",
                                "that's": "that that is", "there'd": "there would",
                                "there'd've": "there would have", "there's": "there is",
                                "they'd": "they would", "they'd've": "they would have",
                                "they'll": "they will", "they'll've": "they will have",
                                "they're": "they are", "they've": "they have",
                                "to've": "to have", "wasn't": "was not", "we'd": "we would",
                                "we'd've": "we would have", "we'll": "we will",
                                "we'll've": "we will have", "we're": "we are",
                                "we've": "we have", "weren't": "were not",
                                "what'll": "what will", "what'll've": "what will have",
                                "what're": "what are", "what's": "what is",
                                "what've": "what have", "when's": "when is",
                                "when've": "when have", "where'd": "where did",
                                "where's": "where is", "where've": "where have",
                                "who'll": "who will", "who'll've": "who will have",
                                "who's": "who is", "who've": "who have",
                                "why's": "why is", "why've": "why have", "will've": "will have",
                                "won't": "will not", "won't've": "will not have",
                                "would've": "would have", "wouldn't": "would not",
                                "wouldn't've": "would not have", "y'all": "you all",
                                "y'all'd": "you all would", "y'all'd've": "you all would have",
                                "y'all're": "you all are", "y'all've": "you all have",
                                "you'd": "you would", "you'd've": "you would have",
                                "you'll": "you will", "you'll've": "you will have",
                                "you're": "you are", "you've": "you have"}
        self.PUNCT_TO_REMOVE = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

    def _clean(self, text):
        text = text.lower()
        # remove html
        regex = re.compile(r'<[^>]+>')
        text = regex.sub('', text)
        # replace_appos
        cleaned_words = []
        for word in text.split():
            if word in self.APPOS_LOWERCASE.keys():
                cleaned_words.append(self.APPOS_LOWERCASE[word])
            else:
                cleaned_words.append(word)
        text = ' '.join(cleaned_words)
        # remove_stopwords
        # text = " ".join([word for word in str(text).split() if word not in self.stoplist])
        # remove punc
        text = text.translate(str.maketrans('', '', self.PUNCT_TO_REMOVE))
        return text

    def _simple_clean(self, text):

        clean_line = ""

        text = text.replace("’", "")
        text = text.replace("'", "")
        text = text.replace("-", " ")  # replace hyphens with spaces
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        text = text.lower()

        for char in text:
            if char in 'qwertyuiopasdfghjklzxcvbnm ':
                clean_line += char
            else:
                clean_line += ' '

        clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
        if len(clean_line) and clean_line[0] == ' ':
            clean_line = clean_line[1:]
        return clean_line

    def __call__(self, texts):
        if isinstance(texts, str):
            return self._simple_clean(texts)
        else:
            return [self._simple_clean(text) for text in texts]


def mark_deleted_words(a, b):
    prev_s = None
    text = ""
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == ' ':
            if prev_s == ' ':
                text += s[-1]
            elif prev_s == '+':
                text += "]" + s[-1]
            elif prev_s == '-':
                text += ")" + s[-1]
            else:
                text += s[-1]
        elif s[0] == '-':
            if prev_s == '-':
                text += s[-1]
            elif prev_s == ' ':
                text += "(" + s[-1]
            elif prev_s == '+':
                text += "] (" + s[-1]
            else:
                text += s[-1]
        elif s[0] == '+':
            if prev_s == '+':
                text += s[-1]
            elif prev_s == ' ':
                text += "[" + s[-1]
            elif prev_s == '+':
                text += ") [" + s[-1]
            else:
                text += s[-1]
        prev_s = s[0]
    return text

def mark_replaced_word(a, b):
    a_words = a.split()
    b_words = b.split()
    result = []
    i = 0
    j = 0
    while i < len(a_words) and j < len(b_words):
        if a_words[i] == b_words[j]:
            result.append(b_words[j])
            i += 1
            j += 1
        else:
            replaced = []
            while j < len(b_words) and b_words[j] not in a_words:
                replaced.append(b_words[j])
                j += 1
            if len(replaced) > 0:
                result.append('【' + " ".join(replaced) + '】')
            else:
                result.append(b_words[j])
                j += 1
    while j < len(b_words):
        result.append(b_words[j])
        j += 1
    return " ".join(result)