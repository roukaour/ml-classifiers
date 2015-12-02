__author__ = 'Jian Yang'
__date__ = '11/12/15'

import copy
#import sklearn.tree
import dtree

import data_smooth
from random import shuffle

# from sklearn.externals.six import StringIO
# import pydot

is_continous = [1 if x in [1, 2, 7, 10, 13, 14] else 0 for x in xrange(15)]

def split_data(data, label):
    num_training = 400
    if True:
        all = [x for x in xrange(600)]
        shuffle(all)
        training_data = [data[all[x]] for x in xrange(num_training)]
        training_label = [label[all[x]] for x in xrange(num_training)]
        test_data = [data[all[x]] for x in xrange(num_training, 600)]
        test_label = [label[all[x]] for x in xrange(num_training, 600)]
    else:
        training_data = data[:num_training]
        test_data = data[num_training:]
        training_label = label[:num_training]
        test_label = label[num_training:]
    return training_data, training_label, test_data, test_label

def get_accuracy(tree, test_data, test_label):
    test_size = len(test_label)

    pred_label = tree.predict(test_data)

    diff = 0
    for i in xrange(test_size):
        if test_label[i] != pred_label[i]:
            diff += 1
    return (test_size-diff) * 1.0 / test_size

def run_test(tree, run_data):
    training_data, training_label, test_data, test_label = run_data

    acc_test = get_accuracy(tree, test_data, test_label)
    acc_train = get_accuracy(tree, training_data, training_label)
    return acc_train, acc_test

def run_method_dtree(run_data):
    training_data, training_label, test_data, test_label = run_data

    tree = dtree.DecisionTree()
    tree.fit(training_data, training_label, is_continous)

    return run_test(tree, run_data)

def draw(clf):
    if False:
        dot_data = StringIO()
        sklearn.tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("iris.pdf")
    else:
        with open("iris.dot", 'w') as f:
            f = sklearn.tree.export_graphviz(clf, out_file=f)

def run_method_sklean(run_data):
    training_data, training_label, test_data, test_label = run_data

    clf = sklearn.tree.DecisionTreeClassifier()
    clf = clf.fit(training_data, training_label)

    run_test(clf, run_data)
    draw(clf)

def predict(data, label):
    training_data, training_label, test_data, test_label = split_data(data, label)

    run_data = [training_data, training_label, test_data, test_label]

    acc_train, acc_test = run_method_dtree(run_data)
    # run_method_sklean(run_data)
    return acc_train, acc_test

def multiple_test(data, label):
    all_train = 0
    all_test = 0
    num_test = 100
    for i in xrange(num_test):
        sub_train, sub_test = predict(data, label)
        all_train += sub_train
        all_test += sub_test
    print all_train * 1.0 / num_test, all_test * 1.0 / num_test

def single_test(data, label):
    acc_train, acc_test = predict(data, label)
    print 'training accuracy is %.2f%%; test accuracy is %.2f%%' % (acc_train*100, acc_test*100)

if __name__ == '__main__':
    data, label = data_smooth.get_data(is_continous)
    # multiple_test(data, label)
    single_test(data, label)
