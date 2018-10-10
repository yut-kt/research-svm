# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_files


def main():
    _, _, tests, _ = load_svmlight_files(['./data/svm_train', './data/svm_test'])
    svc = joblib.load('./model/svc.pkl')
    prediction = svc.predict(tests)

    with open('./data/output', mode='w') as p:
        p.write("\n".join(['1' if result == 1 else '-1' for result in prediction.tolist()]))


if __name__ == '__main__':
    main()
