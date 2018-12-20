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
2--duration
3--主题_one_hot
4--program_type
5-演员_one_hot
6-时间vector # 24
7-时间数
8-星期数
9-创建时间间隔
10-上映时间间隔
11-历史pv
12-pv
"""
def flatten():
    KNN_data = []
    click_data = []

    click_path = os.path.join(os.path.abspath('..'), 'feature_1.0', 'click')
    knn_path = os.path.join(os.path.abspath('..'), 'feature_1.0', 'knn')
    files_click = os.listdir(click_path)
    files_knn = os.listdir(knn_path)

    for file in files_knn:
        print('file ' + file + ' is processing')
        with open(os.path.join(knn_path, file), 'r', encoding='UTF-8') as fread:
            res = fread.read()
            res = json.loads(res)
            for item in res:
                KNN_tmp = [item]
                feature_list = [1, 7, 8, 6, 2, 3, 5, 9, 10]
                for i in feature_list:
                    if isinstance(res[item][i], list):
                        KNN_tmp.extend(res[item][i])
                    else:
                        KNN_tmp.append(res[item][i])
                KNN_data.append(KNN_tmp)
            path = os.path.join(os.path.abspath('..'), 'dataset', 'knn', file)
            numpy.savetxt(path, numpy.array(KNN_data, dtype=int))
            KNN_data.clear()

    for file in files_click:
        print('file ' + file + ' is processing')
        with open(os.path.join(click_path, file), 'r', encoding='UTF-8') as fread:
            res = fread.read()
            res = json.loads(res)
            for item in res:
                tmp = list()
                tmp.append(item)
                tmp.extend(res[item][6])
                tmp.append(res[item][11])
                click_data.append(tmp)
            path = os.path.join(os.path.abspath('..'), 'dataset', 'click', file)
            numpy.savetxt(path, numpy.array(click_data, dtype=int))
            click_data.clear()


if __name__ == '__main__':
    flatten()




