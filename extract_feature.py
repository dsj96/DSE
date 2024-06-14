'''
Descripttion:
version:
Author: ShaojieDai
Date: 2021-05-17 22:47:33
LastEditors: sueRimn
LastEditTime: 2021-05-18 14:28:01
'''
import numpy as np

def get_words(line):
    '''
    @name:
    @msg: to process a line of string
    @param type string eg:  4	Wed Sep 15 21:46:45 +0000 2010	4c2f8b0da0ced13ab4bb106e	{B?nh Cu?n Tay Ho}	{34.081093,-118.084776,San Gabriel,CA,United States}	{Food,}
    @return: <set>  Food
    '''
    words = set()
    line = line.split('\t')
    info = line[-1] # {Performing Arts Venue,Arts & Entertainment,Movie Theater,}\n
    # eg: 6	Sat Mar 26 20:43:37 +0000 2011	4af43cf0f964a520c8f021e3	{10 Bleecker St}	{40.725237,-73.993143,New York,NY,United States}	{}
    word =info.strip("}\n{,").split(',') # 产生空字符串的原因是因为有人没有评论
    for w in word:
        words.add(w)
    return words

def form_dict(f_name):
    '''
    @name: form_dict
    @msg:
    @param {type} string    a filename
    @return: feature type list eg:{'Theme Park', 'Arts & Entertainment', 'Hotel', 'Travel & Transport'}-->list['Arts & Entertainment'...]
    '''
    feature = set()
    # print('We are loading data from:', f_name)
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            words_for_line = get_words(line)
            for word in words_for_line:
                feature.add(word)
    # print(words)
    feature = list(feature)
    # print(len(feature)) # 110
    feature.sort() # 按照字母序排序
    # print(feature)
    return feature


def get_info(feature, f_name):
    '''
    @name:
    @msg:
    @param      feature set经list之后转换的        f_name eg:[ 'Arts', 'Food',...] 已经按字符排序过了
    @return: node_and_feature = {k=str: v=list[int]}  eg: {'4ab5966ff964a5208b7520e3': [1,0,0,0,1...]} 此时只是列表并没有np.array
    '''
    # node_feature = [ 0 for i in range(len(feature))]
    node_and_feature = {}
    # print('We are loading data from:', f_name)
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            node = line.split('\t')[2] # 4ab5966ff964a5208b7520e3
            words_for_line = get_words(line)
            if node in node_and_feature:
                node_feature = node_and_feature[node]
                for w in words_for_line:
                    index = feature.index(w)
                    node_feature[index] = 1

            else:
                node_feature = [ 0 for i in range(len(feature))]
                for w in words_for_line:
                    index = feature.index(w)
                    node_feature[index] = 1
                    node_and_feature[node] = node_feature
    for node,feature in node_and_feature.items():
        node_and_feature[node] = np.array(feature)
    return node_and_feature


def get_user_feature_dict(poi_feature_dict, user_poi_set_dict):
    '''
    @name:
    @msg: 返回U 和 POI_tag 的属性字典
    @param: dict    dict    user_records={u:[poi1, poi2...],...}
    @return: dict
    '''
    # if user_records = {'user_id': set()} ①将具有相同word的次数加和 ②如果 word出现即为1
    # if user_records = {'user_id': list()} ①将具有相同word的次数加和 ②如果 word出现即为1    list 和权重加和的方式好一点
    for k,v in poi_feature_dict.items():
        len_feature = len(poi_feature_dict[k])
        break

    user_feature_dict = {}
    for user, poi_set in user_poi_set_dict.items():
        user_feature_dict[user] = np.array([0. for i in range(len_feature)])
        for poi in poi_set:
            user_feature_dict[user] = user_feature_dict[user] + np.array(poi_feature_dict[poi])
        user_feature_dict[user] = maxmin_norm(user_feature_dict[user])

    return user_feature_dict

def get_user_poi_set(f_name):
    future_history = dict()
    with open(f_name, 'r', encoding= 'utf-8') as f:
        for line in f:
            line = line.strip(' \n').split('\t')
            if line[0] not in future_history.keys():
                future_history[line[0]] = set()
            # future_history[line[0]].add(line[1])
            future_history[line[0]].add(line[2])
    return future_history

def maxmin_norm(x):
    '''
    @name: ShaojieDai
    @Date: 2020-07-20 16:43:38
    @msg: 将一个np.array类型的x ，进行最大最小标准化
    @param {type}
    @return:
    '''
    _range = np.max(x) - np.min(x)
    true_value = (x - np.min(x)) / _range
    round_value = np.round(true_value, decimals=4)
    return round_value

def extract_forsquare_features():
    '''
    @name: ShaojieDai
    @Date: 2021-05-18 14:23:06
    @msg: key=user,poi      value=np.array([]) len=35
    @param {*}
    @return {*}
    '''
    file_path = 'dataset/foursquare/after_removed_all_checkin_file.txt'
    feature = form_dict(file_path)
    poi_feature_dict = get_info(feature, file_path)
    user_poi_set_dict = get_user_poi_set(file_path)
    user_feature_dict = get_user_feature_dict(poi_feature_dict, user_poi_set_dict)
    return poi_feature_dict, user_feature_dict

if __name__ == '__main__':
    file_path = 'dataset/foursquare/after_removed_all_checkin_file.txt'
    feature = form_dict(file_path)
    poi_feature_dict = get_info(feature, file_path)

    user_poi_set_dict = get_user_poi_set(file_path)
    user_feature_dict = get_user_feature_dict(poi_feature_dict, user_poi_set_dict)

    print('over!')