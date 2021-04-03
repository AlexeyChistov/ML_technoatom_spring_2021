import numpy as np


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


def y_predict_conversion(percent, y_predict_):
    """
    Присваиваем элементам y_predict 1 и 0 в зависимости от порога вероятности
    Параллельно проводим валидацию аргумента percent:
        - если percent = None, то threshold = 0.5
        - если percent в диапазоне от 1 до 100, то threshold = percent / 100
        - если percent выходит из диапазона, выбрасываем Exception
    :param percent:
    :param y_predict_:
    :return y_predict_: переделанный y_predict
    """
    if percent is None:
        threshold = .5
    elif (percent >= 1) and (percent <= 100):
        threshold = percent / 100
    else:
        raise Exception("percent should be from 1 to 100 or None")
    y_predict_[y_predict_ >= threshold] = 1
    y_predict_[y_predict_ < threshold] = 0
    return y_predict_


def accuracy_score(y_true, y_predict, percent=None):
    dict_score = dict()
    y_predict_ = y_predict_conversion(percent, y_predict)
    for _ in range(y_predict.shape[1]):
        score = np.mean(y_predict_[:, _] == y_true)
        dict_score[f"class_{_ + 1}"] = score
    return dict_score


def precision_score(y_true, y_predict, percent=None):
    dict_score = dict()
    y_predict_ = y_predict_conversion(percent, y_predict)
    for _ in range(y_predict.shape[1]):
        d = tf_pn(y_true, y_predict_[:, _])
        score = d['TP'] / (d['TP'] + d['FP'])
        dict_score[f"class_{_ + 1}"] = score
    return dict_score


def recall_score(y_true, y_predict, percent=None):
    dict_score = dict()
    y_predict_ = y_predict_conversion(percent, y_predict)
    for _ in range(y_predict.shape[1]):
        d = tf_pn(y_true, y_predict_[:, _])
        score = d['TP'] / (d['TP'] + d['FN'])
        dict_score[f"class_{_ + 1}"] = score
    return dict_score


def lift_score(y_true, y_predict, percent=None):
    dict_score = dict()
    y_predict_ = y_predict_conversion(percent, y_predict)
    for _ in range(y_predict.shape[1]):
        d = tf_pn(y_true, y_predict_[:, _])
        precision = d['TP'] / (d['TP'] + d['FP'])
        score = precision / (d['TP'] + d['FN']) * y_predict.shape[0]
        dict_score[f"class_{_ + 1}"] = score
    return dict_score


def f1_score(y_true, y_predict, percent=None):
    dict_score = dict()
    y_predict_ = y_predict_conversion(percent, y_predict)
    for _ in range(y_predict.shape[1]):
        d = tf_pn(y_true, y_predict_[:, _])
        precision = d['TP'] / (d['TP'] + d['FP'])
        recall = d['TP'] / (d['TP'] + d['FN'])
        score = 2 * precision * recall / (precision + recall)
        dict_score[f"class_{_ + 1}"] = score
    return dict_score


# Отладочные данные
# file = np.loadtxt('HW2_labels.txt',  delimiter=',')
# y_predict, y_true = file[:, :2], file[:, -1]
# print('accuracy', accuracy_score(y_true, y_predict))
# print('precision_score', precision_score(y_true, y_predict))
# print('recall_score', recall_score(y_true, y_predict))
# print('lift_score', lift_score(y_true, y_predict))
# print('f1_score', f1_score(y_true, y_predict))
