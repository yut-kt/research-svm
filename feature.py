# -*- coding: utf-8 -*-

import MeCab
from typing import List, Tuple


class Feature:
    """
    特徴量を扱うクラス
    """

    def __init__(self, feature_path='data/feature.list', tagger=MeCab.Tagger('-Owakati')):
        def get_feature_tuple(feature_line: str) -> Tuple[str, float]:
            _, feature, score = feature_line.split(maxsplit=2)
            return feature, float(score)

        with open(feature_path) as p:
            self.__feature_dict = dict([get_feature_tuple(feature_line) for feature_line in p])
        self.__tagger = tagger

    def convert_sentence(self, sentence: str) -> List[float]:
        """
        文から特徴量へ変換
        :param sentence: 特徴量へ変換したい文
        :return: 特徴量へ変換されたリスト
        """
        return [self.__feature_dict[morpheme] for morpheme in self.__tagger.parse(sentence).split()
                if morpheme in self.__feature_dict]
