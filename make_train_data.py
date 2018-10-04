# -*- coding: utf-8 -*-

import MeCab
import statistics
import numpy


def main():
    features = read_feature_list()
    tagger = MeCab.Tagger("-Owakati")

    trains, labels = [], []
    with open('data/train.list') as p:
        for train_line in p:
            label, _, train_txt = train_line.split(maxsplit=2)
            trains.append(wakati_to_feature_list(tagger.parse(train_txt), features))
            labels.append(label)

    max_width = int(statistics.mean([len(train) for train in trains]) * 2)
    trains = [(train + [0] * (max_width - len(train)))[:max_width] for train in trains]

    assert statistics.mean([len(train) for train in trains]) == max_width, '配列の要素数が次元ごとで揃っていない'
    assert len(trains) == len(labels), '学習データとラベルの数が不一致'

    numpy.savez('data/dataset.npz',
                trains=numpy.array(trains, dtype=numpy.float),
                labels=numpy.array(labels, dtype=numpy.str))


def read_feature_list():
    def get_feature_tuple(feature_line):
        _, feature, score = feature_line.split()
        return feature, float(score)

    with open('data/feature.list') as p:
        return dict([get_feature_tuple(feature_line) for feature_line in p])


def wakati_to_feature_list(wakati, features):
    return [features[morpheme] for morpheme in wakati.split() if morpheme in features]


if __name__ == '__main__':
    main()
