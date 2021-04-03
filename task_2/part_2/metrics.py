import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def tf_pn(y_true: np.ndarray, y_predict: np.ndarray):
    """
    Вспомогательная функция для расчета TP, FP, TN, FN
    :param y_true:
    :param y_predict:
    :return d: словарь из значений TP, FP, TN, FN
    """
    d = dict()
    d['TP'] = np.sum(np.select([y_predict == y_true], [1]) * np.select([y_true == 1], [1]))
    d['FP'] = np.sum(y_predict * np.select([y_predict != y_true], [1]))
    d['TN'] = np.sum(np.select([y_true == 0], [1]) * np.select([y_true == y_predict], [1]))
    d['FN'] = np.sum(np.select([y_predict == 0], [1]) * np.select([y_true != y_predict], [1]))
    return d


def accuracy_score(y_true, y_predict, percent=None):
    score = np.mean(y_predict == y_true)
    print(type(y_true), type(y_predict))
    return score


def precision_score(y_true, y_predict, percent=None):
    d = tf_pn(y_true, y_predict)
    score = d['TP'] / (d['TP'] + d['FP'])
    return score


def recall_score(y_true, y_predict, percent=None):
    d = tf_pn(y_true, y_predict)
    score = d['TP'] / (d['TP'] + d['FN'])
    return score


def lift_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict)
    d = tf_pn(y_true, y_predict)
    score = precision / (d['TP'] + d['FN']) * y_true.shape[0]
    return score


def f1_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    score = 2 * precision * recall / (precision + recall)
    return score


random_state = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=23)
compression_opts = dict(method='zip', archive_name='out.csv')
X_train.to_csv('out.zip', index=False, compression=compression_opts)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
print(y_predict)
y_true = y_test
y_true = np.array(y_true)
print(classification_report(y_true, y_predict))
print('accuracy', accuracy_score(y_true, y_predict))
print('precision_score', precision_score(y_true, y_predict))
print('recall_score', recall_score(y_true, y_predict))
print('lift_score', lift_score(y_true, y_predict))
print('f1_score', f1_score(y_true, y_predict))
