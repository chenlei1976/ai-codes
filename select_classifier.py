#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import metrics
import pickle as pickle
from sklearn.preprocessing import PolynomialFeatures
import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import classifiers
import tools
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import argparse
# from icrawler.builtin import BaiduImageCrawler


def read_data(data_file):
    try:
        data = pd.read_csv(data_file)
    except IOError:
        print('read file [{}] failed!'.format(data_file))
        raise
    seperator = int(len(data) * 0.8)
    train = data[:seperator]
    test = data[seperator:]
    train_y = train.label
    train_x = train.drop('label', axis=1)
    test_y = test.label
    test_x = test.drop('label', axis=1)
    return train_x, train_y, test_x, test_y


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=True,
                    help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
                    help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':

    tools.init_log('./select_classifier.log')

    data_file = "./data.csv"
    thresh = 0.5
    model_save_file = None
    model_save = {}
    random_seed = 22
    test_percentage = 0.2
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
    classifiers = {'NB': classifiers.naive_bayes_classifier,
                   'KNN': classifiers.knn_classifier,
                   'LR': classifiers.logistic_regression_classifier,
                   'RF': classifiers.random_forest_classifier,
                   'DT': classifiers.decision_tree_classifier,
                   'SVM': classifiers.svm_classifier,
                   'SVMCV': classifiers.svm_cross_validation,
                   'GBDT': classifiers.gradient_boosting_classifier
                   }
    need_pipeline = True

    print('reading training and testing data...')
    data_list, target_list = datasets.fetch_flight_vs_receipt()

    X_train, X_test, y_train, y_test = train_test_split(data_list, target_list, test_size=test_percentage,
                                                        random_state=random_seed)

    # pipeline = Pipeline([('vec', CountVectorizer(encoding='cp874', preprocessor=pre_process_1, tokenizer=tokenize_1, stop_words=heavy_st, token_pattern=None)),('tfidf', TfidfTransformer()), ('clf',       MultinomialNB())])
    # parameters = {
    # 'vec__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),
    # 'vec__max_features': (None, 5000, 10000, 20000),
    # 'vec__min_df': (1, 5, 10, 20, 50),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__sublinear_tf': (True, False),
    # 'vec__binary': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
    # }

    # poly2 = PolynomialFeatures(degree=2)
    # poly2.fit_transform(X_train)
    # poly2.transform(X_test)

    count_vec = CountVectorizer()# CountVectorizer(analyzer='word', stop_words='english')
    tfidf_vec = TfidfVectorizer()# TfidfVectorizer(analyzer='word', stop_words='english')

    X_count_train = count_vec.fit_transform(X_train)
    X_count_test = count_vec.transform(X_test)

    parameters = {
        'vect__max_df': (0.5, 0.75), 'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', 'l2')}

    for classifier in test_classifiers:

        logging.info('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](X_count_train, y_train)

        if need_pipeline:

            pipeline_obj = Pipeline([('vect', count_vec), ('tfidf', tfidf_vec), ('clf', model)])
            grid_search = GridSearchCV(pipeline_obj, parameters, n_jobs=1, verbose=1)
            grid_search.fit(X_train, y_train)
            logging.info('training took %fs!' % (time.time() - start_time))
            best_parameters = dict(grid_search.best_estimator_.get_params())
            pipeline_obj.set_params(tfidf__use_idf=best_parameters['tfidf__use_idf'],tfidf__norm=best_parameters['tfidf__norm'],
                                    vect__max_df=best_parameters['vect__max_df'], vect__max_features=best_parameters['vect__max_features'])
            predict = pipeline_obj.predict(X_count_test)
            # if model_save_file is not None:
            #     model_save[classifier] = model
            precision = metrics.precision_score(y_test, predict)
            recall = metrics.recall_score(y_test, predict)
            logging.critical('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
            accuracy = metrics.accuracy_score(y_test, predict)
            logging.critical('accuracy: %.2f%%' % (100 * accuracy))
        else:
            logging.info('training took %fs!' % (time.time() - start_time))
            predict = model.predict(X_count_test)
            if model_save_file is not None:
                model_save[classifier] = model
            precision = metrics.precision_score(y_test, predict)
            recall = metrics.recall_score(y_test, predict)
            logging.critical('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
            accuracy = metrics.accuracy_score(y_test, predict)
            logging.critical('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file is not None:
        pickle.dump(model_save, open(model_save_file, 'wb'))
