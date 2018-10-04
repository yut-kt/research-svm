# -*- coding: utf-8 -*-

import numpy
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def main():
    dataset = numpy.load('data/dataset.npz')
    print(dataset['trains'].shape)
    print(dataset['labels'].shape)

    train_size = int(len(dataset['trains']) * 0.8)
    test_size = len(dataset['trains']) - train_size
    train_data, test_data, train_label, test_label = train_test_split(dataset['trains'],
                                                                                      dataset['labels'],
                                                                                      train_size=train_size,
                                                                                      test_size=test_size)

    tuned_parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]

    score = 'f1'
    clf = GridSearchCV(
        SVC(),
        tuned_parameters,  # 最適化したいパラメータセット
        cv=5,  # 交差検定の回数
        scoring='%s_weighted' % score  # モデルの評価関数の指定
    )
    clf.fit(train_data, train_label)

    print("# Tuning hyper-parameters for %s" % score)
    print()
    print("Best parameters set found on development set: %s" % clf.best_params_)
    print()

    # それぞれのパラメータでの試行結果の表示
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    # テストデータセットでの分類精度を表示
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_label, clf.predict(test_data)
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    main()
