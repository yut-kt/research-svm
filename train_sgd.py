# -*- coding: utf-8 -*-

import numpy
from sklearn import model_selection, svm, metrics, linear_model
from sklearn.externals import joblib


def main():
    dataset = numpy.load('data/dataset.npz')
    print(dataset['trains'].shape)
    print(dataset['labels'].shape)

    # train_size = int(len(dataset['trains']) * 0.8)
    # test_size = len(dataset['trains']) - train_size
    # train_data, test_data, train_label, test_label = model_selection.train_test_split(dataset['trains'],
    #                                                                                   dataset['labels'],
    #                                                                                   train_size=train_size,
    #                                                                                   test_size=test_size)

    sgd = linear_model.SGDClassifier()
    sgd.fit(dataset['trains'], dataset['labels'])
    joblib.dump(sgd, 'model/sgd.pkl')
    # prediction = svc.predict(test_data)
    #
    # ac_score = metrics.accuracy_score(test_label, prediction)
    # print(ac_score)


if __name__ == '__main__':
    main()
