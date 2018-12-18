# -*- encoding: utf-8 -*-
"""
本文件主要是提供根据节目ID查询节目信息的功能
"""


# call the es_search function for the property
# 返回顺序：节目ID, 标题，创建时间，一级分类编号，一级分类名称，剧集类型，剧集时长，节目介绍，关键词，各大属性！
# 剧集类型只有6-9这4种，其中，所有的7和8都有duration，所有9都没有duration，大部分6都有duration --> 过滤掉duration为0的，可以解决没有duration的问题
# 一级分类编号缺失的有3个，其中有两个是一级分类名称缺失，但是这两个的duration都为0，所以实际上只有一条数据有bug --> 过滤掉一级分类编号缺失的3条数据

from urllib import request
import json
import os

LIST = ['1002601', '1002581', '1003861', '1003862', '1003863', '1003864', '1003865', '1003866', '1004041', '1004121',
        '1002781', '1004261', '1004262', '1004281', '1004322', '1004321', '1004301', '1004422', '1004421', '1001381']
def query_first() :
    query_data = {
        "size": 10000,
        "query": {
            "match_all":{}
        }
    }

    return query_data

def query_loop(scroll_id):
    query = {
        "scroll": "1m",
        "scroll_id": scroll_id
    }

    return query

def es_search_full():

    f = open('result.txt', 'w')
    data_first = json.dumps(query_first()).encode('utf-8')

    headers = {
        'content-type': 'application/json'
    }

    req = request.Request("http://10.150.29.111:9200/poms/contents/_search?scroll=10m", data=data_first, headers=headers)

    with request.urlopen(req) as response:
        if response.status==200:
            print('Status:',response.status,response.reason)
            data = json.loads(response.read())
            f.write(str(data['hits']['hits']))
            print(data)
        else:
            raise Exception

    data_second = json.dumps(query_loop(data['_scroll_id'])).encode('utf-8')

    req_second = request.Request("http://10.150.29.111:9200/_search/scroll?", data=data_second, headers=headers)
    with request.urlopen(req_second) as response:
        if response.status==200:
            print('Status:',response.status,response.reason)
            data = json.loads(response.read())
            f.write(str(data['hits']['hits']))
            print(data['_scroll_id'])
        else:
            raise Exception

def es_search():

    data_first = json.dumps(query_first()).encode('utf-8')

    headers = {
        'content-type': 'application/json'
    }

    req_first = request.Request("http://10.150.29.111:9200/poms/contents/_search?scroll=1m", data=data_first, headers=headers)

    response = request.urlopen(req_first)
    if response.status == 200:
        res = json.loads(response.read())
        scroll_id = res['_scroll_id']
        data = res['hits']['hits']
    
    num = 0
    program_information_dir = os.path.join(os.path.abspath('..'), 'program_information')
    count = 0
    result = []
    while len(data):
        for item in data:
            source = item['_source']
            fields = source['fields']

            if source['PRDPACK_ID'] not in LIST:
                continue

            if 'DISPLAYTYPE' not in fields.keys() or fields['DISPLAYTYPE']!='1001':
                continue

            # fix the missed CDuration value
            if 'CDuration' not in fields.keys():
                fields['CDuration']=0

            if 'propertyFileLists' not in fields.keys():
                fields['propertyFileLists'] = {'propertyFile':{}}

            if 'Detail' not in fields.keys():
                fields['Detail'] = ''

            if 'KEYWORDS' not in fields.keys():
                fields['KEYWORDS'] = {'keyword':{}}

            if 'DisplayName' not in fields.keys():
                fields['DisplayName'] = ''
            
            if 'FORMTYPE' not in fields.keys():
                fields['FORMTYPE'] = 0
            
            num = num + 1            

            # fix the keyword
            if isinstance(fields['KEYWORDS']['keyword'], dict):
                if 'keywordName' not in fields['KEYWORDS']['keyword'].keys():
                    fields['KEYWORDS']['keyword']['keywordName'] = ''
                result.append([source['contid'], source['name'], source['createtime'], fields['DISPLAYTYPE'],fields['DisplayName'],fields['FORMTYPE'],
                               fields['CDuration'], fields['Detail'],fields['KEYWORDS']['keyword']['keywordName'], fields['propertyFileLists']['propertyFile']])
            elif isinstance(fields['KEYWORDS']['keyword'], list):
                keyword = []
                for iter in fields['KEYWORDS']['keyword']:
                    if 'keywordName' in iter.keys():
                        keyword.append(iter['keywordName'])
                result.append([source['contid'], source['name'], source['createtime'], fields['DISPLAYTYPE'], fields['DisplayName'],
                               fields['FORMTYPE'], fields['CDuration'], fields['Detail'],keyword, fields['propertyFileLists']['propertyFile']])
            else:
                raise Exception

            if num%10000==0:
                with open(os.path.join(program_information_dir, str(int(num/10000)) + '.txt'), 'w', encoding='UTF-8') as fwrite:
                    print('file ' + str(int(num/10000)) + ' is writing...')
                    res = json.dumps(result, ensure_ascii=False)
                    fwrite.write(res)
                    fwrite.flush()
                    result.clear()

        count = count + 1
        if count%20==0:
            print(num)
        data_loop = json.dumps(query_loop(scroll_id)).encode('utf-8')
        req_loop = request.Request("http://10.150.29.111:9200/_search/scroll?", data=data_loop, headers=headers)
        response = request.urlopen(req_loop)
        if response.status == 200:
            res = json.loads(response.read())
            scroll_id = res['_scroll_id']
            data = res['hits']['hits']
        else:
            raise Exception

    with open(os.path.join(program_information_dir, str(int(num/10000)+1) + '.txt'), 'w', encoding='UTF-8') as fwrite:
        print('file ' + str(int(num/10000)+1) + ' is writing...')
        res = json.dumps(result, ensure_ascii=False)
        fwrite.write(res)
        fwrite.flush()
        result.clear()

if __name__ == '__main__':
    es_search()
