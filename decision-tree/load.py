__author__ = 'Jian Yang'
__date__ = '11/12/15'

import copy
import sklearn.tree
import dtree

is_continous = [1 if x in [1, 2, 7, 10, 13, 14] else 0 for x in xrange(15)]
print is_continous

def load_data():
    data = []
    label = []
    with open("crx.data.txt") as f:
        for l in f:
            elements = l.strip().split(',')
            data.append(elements[:-1])
            label.append(elements[-1])
    return data, label

def flat(data, index):
    if is_continous[index]:
        sum = 0
        num_known = 0
        for i in xrange(len(data)):
            if data[i][index] != '?':
                data[i][index] = int(data[i][index]) if index in [10, 13, 14] else eval(data[i][index])
                sum += data[i][index]
                num_known += 1
        if num_known != len(data):
            average = sum / num_known
            for i in xrange(len(data)):
                if data[i][index] == '?':
                    data[i][index] = average
    else:
        values = {}
        num_values = {}
        for i in xrange(len(data)):
            if data[i][index] != '?':
                if not values.has_key(data[i][index]):
                    num_values[len(values)] = 0
                    values[data[i][index]] = len(values)
                data[i][index] = values[data[i][index]]
                num_values[data[i][index]] += 1
        max_value = [k for k, v in num_values.items() if v == max(num_values.values())][0]
        print max_value, num_values[max_value]
        for i in xrange(len(data)):
            if data[i][index] == '?':
                data[i][index] = max_value

def split_data(data, label):
    training_data = data[:300]
    test_data = data[300:]
    training_label = label[:300]
    test_label = label[300:]
    return training_data, training_label, test_data, test_label


def method1(data, label):
    for i in xrange(15):
        flat(data, i)
    for i in xrange(len(label)):
        if label[i] == '+':
            label[i] = 1
        else:
            label[i] = 0

    training_data, training_label, test_data, test_label = split_data(data, label)

    if True:
        tree = dtree.DecisionTree()
        tree.fit(training_data, training_label, is_continous)
        pred_label = tree.predict(test_data)
    else:
        clf = sklearn.tree.DecisionTreeClassifier()
        clf = clf.fit(training_data, training_label)
        pred_label = clf.predict(test_data)

    diff = 0
    for i in xrange(300):
        print test_label[i], pred_label[i]
        if test_label[i] != pred_label[i]:
            diff += 1
    print diff
    print diff * 1.0 / 300

if __name__ == '__main__':
    data, label = load_data()
    # method 1, assign to majority
    method1(copy.copy(data), label)