# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib


def main():
    trains, labels = load_svmlight_file('data/svm_train')

    svc = SVC()
    svc.fit(trains, labels)
    joblib.dump(svc, 'model/svc.pkl')


if __name__ == '__main__':
    main()
