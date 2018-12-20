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
    target = numpy.loadtxt('../dataset/'+ target_dir)

    data = numpy.array(data, dtype = 'float64')    
    target = numpy.array(target, dtype='int')

    data_max = numpy.max(data, axis=0)#axis=0 -> max value of each column
    data_max[data_max==0]=1
    data_min = numpy.min(data, axis=0)
    data = (data - data_min)/(data_max - data_min)
    data = numpy.nan_to_num(data)

    return data, target

"""
try linear regression again
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


def RFC():
    train_data, train_target = get_data('data.txt', 'target.txt')
    model_rfc = BalancedBaggingClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(), sampling_strategy='auto', replacement=False,random_state=0)
    model_rfc.fit(train_data, train_target)
    save_path_name = '../../model/' + 'rfc.m'
    joblib.dump(model_rfc, save_path_name)


def KNN_train():
    print('knn model is training')
    train_data = numpy.loadtxt('../dataset/'+ 'knn_data.txt')
    train_target = numpy.loadtxt('../dataset/'+ 'knn_target.txt')

    copy = train_data[:, 0:3] #no normalize
    data = train_data[:, 3:]
    data_max = numpy.max(data, axis=0)#axis=0 -> max value of each column
    data_max[data_max==0]=1
    data_min = numpy.min(data, axis=0)
    data = (data - data_min)/(data_max - data_min)
    data = numpy.nan_to_num(data)
    train_data = numpy.hstack((copy, data))

    data_dict = {}
    target_dict = {}
    num = train_data.shape[0]
    for index in range(num):
        item = train_data[index, :]
        key = str(item[0])+'_'+str(item[1])+'_'+str(item[2])
        if key not in data_dict:
            data_dict[key] = [item[3:]]
            target_dict[key] = [[train_target[index]]]
        else:
            data_dict[key].append(item[3:])
            target_dict[key].append(train_target[index])

    model_dict = {}
    for key in data_dict:
        neighbors = 4
        if len(data_dict[key])<4:
            neighbors = len(data_dict[key])
        data = numpy.array(data_dict[key])
        model = KNeighborsRegressor(n_neighbors=neighbors)
        model.fit(numpy.array(data), numpy.array(target_dict[key]))
        model_dict[key]= model
        print(numpy.array(data).shape)

    print('knn model is done')
    save_path = '../../model/'
    for key in model_dict:
        model = model_dict[key]
        name = key+'.m'
        path = os.path.join(save_path, name)
        joblib.dump(model, path)


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
    RFC()
    KNN_train()
    #KNN_predict()
    #Logistic_Regression()
