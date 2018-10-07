# -*- coding: utf-8 -*-

from feature import Feature
import statistics
import numpy


def main():
    feature = Feature()

    trains, labels = [], []
    with open('data/train.list') as p:
        for train_line in p:
            label, _, train_txt = train_line.split(maxsplit=2)
            trains.append(feature.convert_sentence(train_txt))
            labels.append(label)

    max_width = int(statistics.mean([len(train) for train in trains]) * 2)
    trains = [(train + [0] * (max_width - len(train)))[:max_width] for train in trains]

    assert statistics.mean([len(train) for train in trains]) == max_width, '配列の要素数が次元ごとで揃っていない'
    assert len(trains) == len(labels), '学習データとラベルの数が不一致'

    numpy.savez('data/dataset.npz',
                trains=numpy.array(trains, dtype=numpy.float),
                labels=numpy.array(labels, dtype=numpy.str))


if __name__ == '__main__':
    main()
