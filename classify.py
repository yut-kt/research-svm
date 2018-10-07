# -*- coding: utf-8 -*-

from feature import Feature
from sklearn.externals import joblib
import MeCab



def main():
    feature = Feature()

    tests = []
    with open('./data/basis.txt') as p:
        for test_line in p:
            tests.append(feature.convert_sentence(test_line))

    max_width = 35 # TODO: 自動で取れるように
    tests = [(test + [0] * (max_width - len(test)))[:max_width] for test in tests]
    svc = joblib.load('./model/svc.pkl')
    prediction = svc.predict(tests)

    with open('./data/output', mode='w') as p:
        p.write("\n".join(['1' if result == '+1' else '-1' for result in  prediction.tolist()]))


def read_feature_list():
    def get_feature_tuple(feature_line):
        _, feature, score = feature_line.split()
        return feature, float(score)

    with open('feature.list') as p:
        return dict([get_feature_tuple(feature_line) for feature_line in p])


def wakati_to_feature_list(wakati, features):
    return [features[morpheme] for morpheme in wakati.split() if morpheme in features]


if __name__ == '__main__':
    main()
