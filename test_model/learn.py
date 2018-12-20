# -*- encoding: utf-8 -*-
"""
本文件主要有以下几个作用：
1 从dataset下读取数据x和y，进行简单的预处理后，返回可用于直接训练的x和y
2. 使用不同的模型对x和y进行交叉验证
"""

import json
import os
import numpy
import random
import imp
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, classification_report, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from tensorflow import keras
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor

"""
获取数据x和y
"""
def rfc_get_data():
    path = os.path.join(os.path.abspath('..'), 'dataset', 'click')
    files = os.listdir(path)

    res_path = os.path.join(os.path.abspath('../..'), 'predict')

    for file in files:
        data = numpy.loadtxt(os.path.join(path, file))
        res = rfc_predict(data)
        fwrite = open(os.path.join(res_path, file), 'w', encoding='utf-8')
        fwrite.write(json.dumps(res, ensure_ascii=False))

def rfc_predict(data):
    res = data[:, 0:1]
    ID = data[:, 0:25]
    input = data[:, 25:]
    input_max = numpy.max(input, axis=0)  # axis=0 -> max value of each column
    input_max[input_max == 0] = 1
    input_min = numpy.min(input, axis=0)
    input = (input - input_min) / (input_max - input_min)
    input = numpy.nan_to_num(input)

    data = numpy.hstack((ID, input))

    save_path = '../../model/rfc.m'
    model_rfc = joblib.load(save_path)

    predicted = model_rfc.predict_proba(data[:, 1:])
    predicted_target = numpy.argmax(predicted, axis=1)
    predicted_prob = numpy.max(predicted, axis=1)
    target = predicted_target.reshape(-1, 1)
    prob = predicted_prob.reshape(-1, 1)

    res = numpy.hstack((res, target))
    res = numpy.hstack((res, prob))
    # return ID + target + probability, for sorting

    res = res.tolist()
    return res

'''
1+7+8(3) + 6(24) + other features for knn
'''
def cold_start():
    print('knn model is running')

    # load the knn model into dict
    model_path = '../../model/'
    models = os.listdir(model_path)
    model_dict = {}
    for model in models:
        if 'knn' in model:
            continue
        key = model.split('.m')[0]
        tmp_model = joblib.load(os.path.join(model_path, model))
        model_dict[key] = tmp_model

    path = os.path.join(os.path.abspath('..'), 'dataset', 'knn')
    files = os.listdir(path)

    res_path = os.path.join(os.path.abspath('../..'), 'predict')

    for file in files:
        data = numpy.loadtxt(os.path.join(path, file))
        knn_key = data[:, 0:28]
        knn_data = data[:, 28:]
        data_max = numpy.max(knn_data, axis=0)
        data_max[data_max == 0] = 1
        data_min = numpy.min(knn_data, axis=0)
        knn_data = (knn_data - data_min)/(data_max - data_min)
        knn_data = numpy.nan_to_num(knn_data)
        data = numpy.hstack((knn_key, knn_data))
 
        data_dict = {}
        num = data.shape[0]

        for index in range(num):
            item = data[index, :]
            key = str(item[1]) + '_' + str(item[2]) + '_' + str(item[3])
            if key not in data_dict:
                data_dict[key] = []
            tmp = [item[0]]
            tmp.extend(item[4:])
            data_dict[key].append(tmp)

        data = []
        for key in data_dict:
            value = numpy.array(data_dict[key])
            if key in model_dict:
                model = model_dict[key]
                print(value[:, 25:].shape)
                tmp = model.predict(value[:, 25:])

                tmp = tmp.reshape(-1, 1)
                tmp_data = value[:, :25]
                tmp_data = numpy.hstack((tmp_data, tmp))
                data.extend(tmp_data)
            else:
                tmp = numpy.array([random.randint(1,10) for i in range(value.shape[0])])
                tmp_data = value[:, :25]
                tmp = tmp.reshape(-1,1)
                print(tmp.shape)
                print(tmp_data.shape)
                tmp_data = numpy.hstack((tmp_data, tmp))
                data.extend(tmp_data)

        data = numpy.array(data)
        res = rfc_predict(data)
        fwrite = open(os.path.join(res_path, file), 'w', encoding='utf-8')
        fwrite.write(json.dumps(res, ensure_ascii=False))

        # code for saving the result
    print('knn predict is done')


if __name__ == '__main__':
    rfc_get_data()
    cold_start()
