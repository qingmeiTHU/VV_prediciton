# -*- encoding: utf-8 -*-
"""
本文件主要有以下几个作用：
1 从feature_1.0下读取数据，根据要求制作数据集x和y，存放到dataset文件夹下
"""

import os
import json
import math
import numpy
import imp
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from collections import Counter
"""
数据过滤条件，可自定义，只要目标数据在此函数中返回True即可
"""

def judge(PV):
    if int(PV) < 10:
        return False
    return True

"""
pv到类别的对应关系，可自定义
0.1 0.2 4 18 78
"""
def get_class(pv_str):
    pv = int(pv_str)
    
    if pv<10:
        return 0
    elif pv < 100:
        return 1
    elif pv < 500:
        return 2
    elif pv < 1000:
        return 3
    else:
        return 4


"""
生成数据x和y
file_list：要遍历的特征文件的list，可以选取1-21中的任意多个文件
feature_list：要使用的特征索引的list，一共12个特征，特征索引（0-11）和特征名字的对应关系如下
0--displaytype
1--formtype
2--duration #1
3--主题_one_hot #
4--program_type
5-演员_one_hot #300
6-时间vector # 24
7-时间数
8-星期数
9-创建时间间隔 #1
10-上映时间间隔 #1
11-历史pv
12-pv
"""
def flatten():
    flatten_data = []
    flatten_target = []
    KNN_data = []
    KNN_target = []

    count = 0

    path = os.path.join(os.path.abspath('..'), 'feature_1.0')
    files = os.listdir(path)
    for file in files:
        print('file ' + file + ' is processing')
        with open(os.path.join(os.path.abspath('..'), 'feature_1.0', file), 'r', encoding='UTF-8') as fread:
            res = fread.read()
            res = json.loads(res)
            
            for item in res:
                if int(res[item][0])!= 1:
                    continue
                count = count + 1
                tmp = []
                tmp.extend(res[item][6])
                tmp.append(res[item][11])
                flatten_data.append(tmp)
                flatten_target.append(int(res[item][12]))
                KNN_tmp = []
                feature_list = [1,7,8,2,3,5,9,10]
                for i in feature_list:
                    if isinstance(res[item][i], list):
                        KNN_tmp.extend(res[item][i])
                    else:
                        KNN_tmp.append(res[item][i])
                KNN_data.append(KNN_tmp)
                KNN_target.append(int(res[item][11]))

    target = list(flatten_target)
    target.sort(reverse=True)
    level1 = int(len(target)*0.001)
    level2 = int(len(target)*0.003)
    level3 = int(len(target)*0.043)
    level4 = int(len(target)*0.243)
    print(str(target[0]) + ' ' + str(target[len(target)-1]))
    print(str(target[level1]) + ' ' + str(target[level2]) + ' ' + str(target[level3]) + ' ' + str(target[level4]))
    for i in range(len(flatten_target)):
        if flatten_target[i] < target[level4]:
            flatten_target[i] = 0
        elif flatten_target[i] < target[level3]:
            flatten_target[i] = 1
        elif flatten_target[i] < target[level2]:
            flatten_target[i] = 2
        elif flatten_target[i] < target[level1]:
            flatten_target[i] = 3
        else:
            flatten_target[i] = 4

    flatten_data = numpy.array(flatten_data, dtype = 'float64')
    KNN_data = numpy.array(KNN_data, dtype='float64')
    KNN_target = numpy.array(KNN_target, dtype='int')
    print(Counter(flatten_target))
    print(count)
    train_data_path = os.path.join(os.path.abspath('..'), 'dataset', 'data.txt')
    numpy.savetxt(train_data_path, flatten_data)
    train_target_path = os.path.join(os.path.abspath('..'), 'dataset', 'target.txt')
    numpy.savetxt(train_target_path, flatten_target)

    knn_train_data_path = os.path.join(os.path.abspath('..'), 'dataset', 'knn_data.txt')
    numpy.savetxt(knn_train_data_path, KNN_data)
    knn_train_target_path = os.path.join(os.path.abspath('..'), 'dataset', 'knn_target.txt')
    numpy.savetxt(knn_train_target_path, KNN_target)

if __name__ == '__main__':
    flatten()




