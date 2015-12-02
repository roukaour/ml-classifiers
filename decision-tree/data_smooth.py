__author__ = 'Jian Yang'
__date__ = '11/30/15'

from collections import Counter

def get_eval(x):
    if x == '0':
        return 0
    if x[0] == '0':
        return get_eval(x[1:])
    return eval(x)

def get_cate(feat_table, element):
    if not feat_table.has_key(element):
        feat_table[element] = len(feat_table)
    return feat_table[element]

def normalize(feature_table, element, is_continous):
    if element == '?':
        return None
    if is_continous:
        return get_eval(element)
    else:
        return get_cate(feature_table, element)

def load_data(is_continous):
    data = []
    label = []
    feature_table = {x:{} for x in range(len(is_continous))}
    with open("crx.data.txt") as f:
        for l in f:
            elements = l.strip().split(',')
            assert len(elements) == len(is_continous)+1
            features = [normalize(feature_table[i], elements[i], is_continous[i]) for i in xrange(len(is_continous))]
            data.append(features)
            label.append(1 if elements[-1] == '+' else 0)

    return data, label

def smooth_continous(data, index):
    # use average
    not_none = [data[x][index] for x in xrange(len(data)) if data[x][index] is not None]
    aver = sum(not_none) / len(not_none)
    for i in xrange(len(data)):
        if data[i][index] is None:
            data[i][index] = aver

def smooth_discrete(data, index):
    # use majority
    cnts = Counter([x[index] for x in data if x[index] is not None])
    majority = cnts.most_common(1)[0][0]

    for i in xrange(len(data)):
        if data[i][index] is None:
            data[i][index] = majority

def data_smooth(data, is_continous):
    for i in xrange(15):
        if is_continous[i]:
            smooth_continous(data, i)
        else:
            smooth_discrete(data, i)

def get_data(is_continous):
    data, label = load_data(is_continous)
    data_smooth(data, is_continous)
    return data, label

if __name__ == '__main__':
    import copy
    # method 1, assign to majority
    is_continous = [1 if x in [1, 2, 7, 10, 13, 14] else 0 for x in xrange(15)]
    data, label = load_data(is_continous)
    data_smooth(data, is_continous)