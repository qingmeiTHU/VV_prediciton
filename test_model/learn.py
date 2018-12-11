# -*- encoding: utf-8 -*-
"""
本文件主要有以下几个作用：
1 从dataset下读取数据x和y，进行简单的预处理后，返回可用于直接训练的x和y
2. 使用不同的模型对x和y进行交叉验证
"""

import json
import os
import numpy
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
def get_data(data_dir, target_dir):
    data = []
    target = []
    data = numpy.loadtxt('../dataset/'+ data_dir)
    
    data = numpy.array(data, dtype = 'float64')
    target = numpy.array(target, dtype='int')

    data_max = numpy.max(data, axis=0)#axis=0 -> max value of each column
    data_max[data_max==0]=1
    data_min = numpy.min(data, axis=0)
    data = (data - data_min)/(data_max - data_min)
    data = numpy.nan_to_num(data)

    return data

"""
try regression again 
"""
def Linear_Regression():
    test_data, test_target, train_data, train_target = get_data()
    model_LinearRegression = Lasso(alpha=0.1)
    model_LinearRegression.fit(train_data, train_target)
    predicted = model_LinearRegression.predict(test_data)
    print(mean_squared_error(predicted, test_target)) 
    print(predicted.tolist()[:30])
    print(test_target.tolist()[:30])
    # save_path_name = ''
    #joblib.dump(model, save_path_name)

"""
逻辑斯蒂分类
"""
def Logistic_Regression():
    test_data, test_target, train_data, train_target = get_data()
    model_LogisticRegression = LogisticRegression()
    model_LogisticRegression.fit(train_data,train_target)
    print(model_LogisticRegression.coef_)
    predicted_target = model_LogisticRegression.predict(test_data)
    print(classification_report(test_target, predicted_target)) 


def RFC_predict():
    save_path = '../../model/rfc.m'
    data = get_data('data.txt')
    model_rfc = joblib.load(save_path)
    predicted = model_rfc.predict_proba(data)
    predicted_target = numpy.argmax(predicted, axis=1)
    predicted_prob = numpy.max(predicted, axis=1)
    target = predicted_target.reshape(-1, 1)
    prob = predicted_prob.reshape(-1, 1)
    # return ID + target + probability, for recommendation

'''
ID + 6 + other features for knn
'''
def KNN_predict():
    print('knn model is testing')
    data = numpy.loadtxt('../dataset/'+ 'knn_data.txt')

    # load the model into dict
    model_path = '../../model/'
    models = os.listdir(model_path)
    model_dict = {}
    for model in models:
        if 'knn' in model:
            continue
        key = model.split('.m')[0]
        tmp_model = joblib.load(os.path.join(model_path, model))
        model_dict[key] = tmp_model

    predict_target = []
    data_dict = {}
    num = data.shape[0]
    for index in range(num):
        item = data[index, :]
        key = str(item[0])+'_'+str(item[1])+'_'+str(item[2])
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(item[3:])

    data = []
    predicted = []
    for key in data_dict:
        if key in model_dict:
            model = model_dict[key]
            value = numpy.array(data_dict[key])
            tmp = model.predict(value[:, 26:])

            tmp = tmp.reshape(-1, 1)
            tmp_data = value[:, :24]
            tmp_data = numpy.hstack((tmp_data, tmp))
            data.extend(tmp_data)
        else:
            tmp = numpy.array([[5] for i in range(value.shape[0])])
            tmp_data = value[:, :24]
            tmp_data = numpy.hstack((tmp_data, tmp))

            data.extend(tmp_data)

    data = numpy.array(data)
    print('knn predict is done')
    return data


def coldstart():
    print('cold start is training')
    knn_data = KNN_predict()

    data_max = numpy.max(knn_data, axis=0)  # axis=0 -> max value of each column
    data_max[data_max == 0] = 1
    data_min = numpy.min(knn_data, axis=0)
    data = (knn_data - data_min) / (data_max - data_min)
    knn_data = numpy.nan_to_num(data)

    save_path = '../../model/rfc.m'
    model_rfc = joblib.load(save_path)

    predicted = model_rfc.predict_proba(knn_data)

    predicted_target = numpy.argmax(predicted, axis=1)
    predicted_prob = numpy.max(predicted, axis=1)
    target = predicted_target.reshape(-1, 1)
    prob = predicted_prob.reshape(-1, 1)


def nn():
    test_data, test_target, train_data, train_target = get_data()
    target = keras.utils.to_categorical(train_target)
    model = keras.Sequential()
    model.add(keras.layers.Dense(300, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(40, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, target,epochs=20,batch_size=100)
    predicted = model.predict(test_data)
    proba = numpy.argmax(predicted, axis=1)
    #label = numpy.where(proba=predicted)
    print(classification_report(test_target, proba))
    #print(proba)

if __name__ == '__main__':
    RFC_predict()
    coldstart()
