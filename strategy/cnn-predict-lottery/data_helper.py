# from urllib import request, parse
import json
import numpy as np

_showapi_appid = '43943'
_showapi_sign = 'cb8ab2087d7a42e7b2e1a9470659bc42'

url = 'http://route.showapi.com/44-2'


def randomOpenCode(num):
    training_set = []
    for i in range(num):
        code = np.random.choice(32, 5, replace=False)
        code = np.append(code, np.random.choice(32, 1, replace=False))
        training_set.append(code)

    return np.array(training_set)

def convertDataToY(lists):
    res = []
    for now in lists:
        tmp = np.zeros(32)
        for i in now:
            tmp[i] = 1
        res.append(tmp)
    return np.array(res)

def convertYToData(lists):
    res = []
    for now in lists:
        sort = now.copy()
        print('sort:', sort)
        tmp = []
        tmp = np.argpartition(sort, 6)
        print('tmp:', tmp)
        # for i in now:
        #     if(i )

list = randomOpenCode(2)
print('list:', list)
after = convertDataToY(list)
print('after:', after)
back = convertYToData(after)
print('back:', back)

# def getData():
#     send_data = parse.urlencode([
#         ('showapi_appid', _showapi_appid),
#         ('showapi_sign', _showapi_sign),
#         ("code", "ssq"),
#         ("count", 50)
#
#     ])
#
#     req = request.Request(url)
#     try:
#         response = request.urlopen(req, data=send_data.encode('utf-8'), timeout=10)
#     except Exception as e:
#         print(e)
#
#     result = response.read().decode('utf-8')
#     result_json = json.loads(result)
#     print('result_json data is:', result_json)
#
#     results = result_json['showapi_res_body']['result']
#
#     train_examples = []
#     for res in results:
#         tmp = str.split(res['openCode'], ',')
#         last_index = len(tmp) - 1
#
#         print('tmp', tmp, 'len[tmp] - 1', last_index)
#         last_codes = str.split(tmp[last_index], '+')
#         tmp[last_index] = last_codes[0]
#         tmp.append(last_codes[1])
#         train_examples.append(np.array(tmp).astype(int))
#
#     print('result_json data is:', len(results))
#     print('train_examples is:', train_examples)
#
#     return np.array(train_examples)

