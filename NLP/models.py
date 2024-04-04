# MIT License

# Copyright (c) 2023 Fatima El Fouladi, Anum Siddiqui, Jeff Wilgus, David Lemme, Mira Aji, Adam Qamar, Shabia Saeed, Raya Maria Lahoud , Nelly Bozorgzad, Joshua-James Nantel-Ouimet .

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from joblib import dump
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from logger_config import configure_logger
logger = configure_logger(__name__)

def train_random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    kf = KFold(n_splits=6)
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        logger.info(f"RandomForest F1 Score: {f1}")
    average_f1_score = np.mean(f1_scores)
    logger.info(f"RF Average F1 Score: {average_f1_score}")
    return clf, average_f1_score

def train_xgboost(X, y, num_classes):
    clf_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)
    kf = KFold(n_splits=6)
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_xgb.fit(X_train, y_train)
        y_pred = clf_xgb.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        logger.info(f"XGBoost F1 Score: {f1}")
    average_f1_score = np.mean(f1_scores)
    logger.info(f"XGB Average F1 Score: {average_f1_score}")
    return clf_xgb, average_f1_score

def train_naive_bayes(X, y):
    clf_nb = MultinomialNB()
    kf = KFold(n_splits=6)
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_nb.fit(X_train, y_train)
        y_pred = clf_nb.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        logger.info(f"Naive Bayes F1 Score: {f1}")
    average_f1_score = np.mean(f1_scores)
    logger.info(f"NB Average F1 Score: {average_f1_score}")
    return clf_nb, average_f1_score

def train_logistic_regression(X, y):
    clf_lr = LogisticRegression(random_state=42, max_iter=1000)
    kf = KFold(n_splits=6)
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_lr.fit(X_train, y_train)
        y_pred = clf_lr.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        logger.info(f"Logistic Regression F1 Score: {f1}")
    average_f1_score = np.mean(f1_scores)
    logger.info(f"LR Average F1 Score: {average_f1_score}")
    return clf_lr, average_f1_score

def train_svm(X, y):
    clf_svm = SVC(random_state=42)
    kf = KFold(n_splits=6)
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_svm.fit(X_train, y_train)
        y_pred = clf_svm.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
        logger.info(f"SVM F1 Score: {f1}")
    average_f1_score = np.mean(f1_scores)
    logger.info(f"SVM Average F1 Score: {average_f1_score}")
    return clf_svm, average_f1_score


def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["tweet"])
    y = df["class"]
    return X, y, vectorizer

def vectorize_emotion_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["tweet"])
    y = df["label"]
    return X, y, vectorizer
