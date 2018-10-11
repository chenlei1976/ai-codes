
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.model_selection import GridSearchCV


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y, is_default=True):
    from sklearn.ensemble import RandomForestClassifier

    if is_default:
        model = RandomForestClassifier()
        model.fit(train_x, train_y)
        return model
    else:
        params = {
            'n_estimators': range(80, 120, 10),
            'max_features': np.linspace(0.5, 0.9, num=5).tolist(),
            'max_depth': range(3, 9) + [None],
        }
        grid = GridSearchCV(RandomForestClassifier(), params, cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(train_x, train_y)
        return RandomForestClassifier(**grid.best_params)


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y, is_default=True):
    from sklearn.ensemble import GradientBoostingClassifier
    if is_default:
        params = {
            'n_estimators': range(80, 120, 10),
            'max_features': range(0.6, 0.9, 0.1).tolist(),
            'max_depth': range(3, 9) + [None],
        }
        grid = GridSearchCV(GradientBoostingClassifier(), params, cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(train_x, train_y)
        return GradientBoostingClassifier(**grid.best_params)
    else:
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(probability=True)

    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2], 'gamma': [0.125, 0.5]}

    param_grid = {'kernel': ('linear', 'rbf'), 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
    model = SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'],
                probability=True)
    model.fit(train_x, train_y)
    return model
