# -*- encoding: utf-8 -*-
"""
本文件主要有以下几个作用：
1 将dat_1.0下的数据文件和feature下的特征文件进行拼接，对应存放到feature_1.0下
"""

from extra_feature import process_time, calculate_releasetime_interval, calculate_createtime_interval
import os
import json
import math
import numpy

def click_dict():
    click = {}
    time = ''
    path = os.path.join(os.path.abspath('..'), 'dat_2.0')
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(os.path.abspath('..'), 'dat_2.0', file), 'r', encoding='UTF-8') as fread:
            for line in fread.readlines():
                line = line.replace('\r', '').replace('\n', '')
                tmp = line.split('|')
                click[tmp[1]] = int(tmp[2])
                time = tmp[0]

    return click, time

"""
将特征和数据进行拼接，feature_1.0中的格式如下
Key:
    time_contentID
Value(13):
    displaytype
    formtype
    duration
    主题_one_hot
    program_type
    演员_one_hot
    时间vector
    时间数
    星期数
    创建时间间隔
    上映时间间隔
    历史pv
    PV值
"""
def concate_to_feature_1():
    path = os.path.join(os.path.abspath('..'), 'feature')
    files = os.listdir(path)

    click, time = click_dict()

    click_data = {}

    for file in files:
        print('file ' + file + ' is processing')
        dataset = {}

        with open(os.path.join(os.path.abspath('..'), 'feature', file), 'r', encoding='UTF-8') as fread_feature, \
                open(os.path.join(os.path.abspath('..'), 'feature_1.0', 'knn', file), 'w', encoding='UTF-8') as fwrite:

            feature_dict = fread_feature.read()
            feature_dict = json.loads(feature_dict)

            for key in feature_dict:
                Item = feature_dict[key]
                time_vector, time_num, weekday_num = process_time(time)
                releasetime_interval = calculate_releasetime_interval(time, Item[4])
                createtime_interval = calculate_createtime_interval(time, Item[0])
                tmp = []
                tmp.extend(Item[1:4])
                tmp.extend(Item[5:7])
                tmp.append(Item[7][0:300])
                tmp.extend([time_vector, time_num, weekday_num, createtime_interval, releasetime_interval])
                if key in click:
                    tmp.append(click[key])
                    click_data[key] = tmp
                else:
                    dataset[key] = tmp

            fwrite.write(json.dumps(dataset, ensure_ascii=False))
            dataset.clear()

    with open(os.path.join(os.path.abspath('..'), 'feature_1.0', 'click', 'data.txt'), 'w', encoding='UTF-8') as fwrite:
        fwrite.write(json.dumps(click_data, ensure_ascii=False))


if __name__ == '__main__':
    concate_to_feature_1()
