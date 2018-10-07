# -*- coding: utf-8 -*-

import numpy
from sklearn.svm import SVC
from sklearn.externals import joblib


def main():
    dataset = numpy.load('data/dataset.npz')

    svc = SVC()
    svc.fit(dataset['trains'], dataset['labels'])
    joblib.dump(svc, 'model/svc.pkl')


if __name__ == '__main__':
    main()
