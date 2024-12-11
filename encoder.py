import re
from abc import ABCMeta, abstractmethod
from typing import Optional, Set, Dict, List


class StrEncoder(metaclass=ABCMeta):
    _vocabulary: Dict[str, int] = {}

    def _get_or_create(self, key: str) -> int:
        result: Optional[int] = self._vocabulary.get(key)

        if result is not None:
            return result

        result = len(self._vocabulary)
        self._vocabulary[key] = result
        return result

    def vocabulary_size(self):
        return len(self._vocabulary)

    @abstractmethod
    def encode(self, value: str) -> Set[int]:
        ...


class ShingleEncoder(StrEncoder):
    _vocabulary: Dict[str, int] = {}
    _k: int

    def __init__(self, k: int):
        super().__init__()
        self._k = k

    def encode(self, value: str) -> Set[int]:
        k = self._k

        if len(value) < k:
            return set()
        elif len(value) == k:
            return {self._get_or_create(value)}

        return {self._get_or_create(value[i:(i + k)]) for i in range(len(value) - k + 1)}

    @staticmethod
    def shingle(value: str, k: int) -> Set[str]:
        if len(value) < k:
            return set()
        elif len(value) == k:
            return {value}

        return {value[i:(i + k)] for i in range(len(value) - k + 1)}


class ModelWordsEncoder(StrEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, value: str) -> Set[int]:
        matches = ModelWordsEncoder.get_model_words(value)

        return {self._get_or_create(v) for v in matches}

    @staticmethod
    def get_model_words(value: str) -> List[str]:
        return re.findall('[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*', value)
    
class DomainFeaturesEncoder(StrEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, value: Dict[str, List[int]]) -> Set[int]:
        feature_pairs = DomainFeaturesEncoder.get_feature_pairs(value)

        return {self._get_or_create(fp) for fp in feature_pairs}

    @staticmethod
    def get_feature_pairs(value: Dict[str, List[int]]) -> List[str]:

        feature_strings = []

        for key, values in value.items():
            for value in values:
                feature_strings.append(f"{key}_{value}")

        return feature_strings
    
class PlusFeaturesEncoder(StrEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, value: Dict[str, List[int]]) -> Set[int]:
        feature_values = PlusFeaturesEncoder.get_feature_values(value)

        return {self._get_or_create(fv) for fv in feature_values}

    @staticmethod
    def get_feature_values(value: Dict[str, List[int]]) -> List[str]:

        feature_values = []

        for key, values in value.items():
            for value in values:
                feature_values.extend(re.findall('ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$', value))

        return feature_values
