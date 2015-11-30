__author__ = 'Jian Yang'
__date__ = '11/21/15'

import math
import copy

def getlog(num, total):
    ratio = num * 1.0 / total
    if ratio == 0:
        return 0
    assert ratio >= 0
    return ratio * math.log(ratio, 2)

class Data:
    def __init__(self, tdata, tlabel, continous):
        self.tdata = tdata
        self.tlabel = tlabel
        self.continous = continous

class Tree:
    def __init__(self, data, valid_data, valid_feature):
        self.data = data
        self.valid_data = valid_data
        self.valid_features = valid_feature

        self.arbitor = None
        self.trees = {}
        self.feature_value = None

    def check(self):
        is_all_1 = True
        is_all_0 = True
        for i in self.valid_data:
            if self.data.tlabel[i] == 1:
                is_all_0 = False
            else:
                is_all_1 = False
            if not is_all_0 and not is_all_1:
                return is_all_1, is_all_0
        return is_all_1, is_all_0

    def majarity_1(self):
        num_1 = 0
        num_0 = 0
        for i in self.valid_data:
            if self.data.tlabel[i] == 1:
                num_1 += 1
            else:
                num_0 += 1
        return True if num_1 >= num_0 else False

    def get_entropy_countinues(self, feature):
        f = feature
        all_values = [self.data.tdata[x][f] for x in self.valid_data]
        all_values = set(all_values)

        max_entropy = 0
        max_value = None
        for sub_value in all_values:
            num_le = 0
            num_gt = 0
            num_l0 = 0
            num_l1 = 0
            num_g0 = 0
            num_g1 = 0
            for d in self.valid_data:
                if self.data.tdata[d][f] <= sub_value:
                    num_le += 1
                    if self.data.tlabel[d] == 1:
                        num_l1 += 1
                    else:
                        num_l0 += 1
                else:
                    num_gt += 1
                    if self.data.tlabel[d] == 1:
                        num_g1 += 1
                    else:
                        num_g0 += 1
            if num_gt == 0:
                continue
            sub_entropy = (-getlog(num_l0, num_le)-getlog(num_l1, num_le)) * (num_le * 1.0 / (num_le + num_gt)) + \
                          (-getlog(num_g0, num_gt)-getlog(num_g1, num_gt)) * (num_gt * 1.0 / (num_le + num_gt))

            if sub_entropy > max_entropy:
                max_entropy = sub_entropy
                max_value = sub_value

        return max_entropy, max_value

    def get_entropy_discrete(self, feature):
        f = feature
        values = {}
        for d in self.valid_data:
            if not values.has_key(self.data.tdata[d][f]):
                values[self.data.tdata[d][f]] = 0
            values[self.data.tdata[d][f]] += 1
        total = sum(values.values())
        assert total
        entropy = 0
        for v, num_v in values.items():
            num_1 = 0
            num_0 = 0
            for d in self.valid_data:
                if self.data.tdata[d][f] == v:
                    if self.data.tlabel[d] == 1:
                        num_1 += 1
                    else:
                        num_0 += 1
            sub_entropy = -getlog(num_0, num_v) -getlog(num_1, num_v)
            entropy += sub_entropy * num_v * 1.0 / total
        return entropy

    def split(self, feature, split_value): # split_value is for countinous feature only
        f = feature
        self.feature_num = f

        if self.data.continous[f]:
            self.feature_value = split_value

            le_data = []
            gt_data = []

            le_value = set()
            gt_value = set()
            for d in self.valid_data:
                if self.data.tdata[d][f] <= self.feature_value:
                    le_value.add(self.data.tdata[d][f])
                    le_data.append(d)
                else:
                    gt_value.add(self.data.tdata[d][f])
                    gt_data.append(d)

            gt_features = self.valid_features
            le_features = self.valid_features
            if len(gt_value) == 1:
                gt_features = copy.copy(self.valid_features)
                gt_features.remove(f)

            if len(le_value) == 1:
                le_features = copy.copy(self.valid_features)
                le_features.remove(f)

            assert len(le_data) + len(gt_data) == len(self.valid_data)
            print len(le_data), len(gt_data)

            if len(le_data) != 0 and len(gt_data) != 0:
                self.trees['le'] = Tree(self.data, le_data, le_features)
                self.trees['gt'] = Tree(self.data, gt_data, gt_features)

                self.trees['le'].fit()
                self.trees['gt'].fit()
            else:
                self.arbitor = self.majarity_1()

        else:
            values = set([self.data.tdata[x][f] for x in self.valid_data])
            sub_features = copy.copy(self.valid_features)
            sub_features.remove(f)
            for sub_value in values:
                sub_data = [x for x in self.valid_data if self.data.tdata[x][f] == sub_value]

                self.trees[sub_value] = Tree(self.data, sub_data, sub_features)
                self.trees[sub_value].fit()

    def get_feature(self):
        entropy_feature = {}
        cont_feature = {}
        for f in self.valid_features:
            assert f < len(self.data.continous), f
            if self.data.continous[f]: # continous
                max_entropy, max_value = self.get_entropy_countinues(f)
                entropy_feature[f] = max_entropy
                cont_feature[f] = max_value
            else:
                entropy_feature[f] = self.get_entropy_discrete(f)
        print entropy_feature
        print cont_feature

        max_e = max(entropy_feature.values())
        for f, v in entropy_feature.items():
            if v == max_e:
                self.split(f, cont_feature[f] if cont_feature.has_key(f) else None)
                break

    def is_continous(self):
        return self.feature_value is not None

    def predict(self, sdata):
        if self.arbitor is not None:
            return 1 if self.arbitor else 0
        if self.is_continous():
            if sdata[self.feature_num] <= self.feature_value:
                return self.trees['le'].predict(sdata)
            else:
                return self.trees['gt'].predict(sdata)
        else:
            assert isinstance(self.feature_num, int), self.feature_num
            if self.trees.has_key(sdata[self.feature_num]):
                return self.trees[sdata[self.feature_num]].predict(sdata)
            else:
                return 0

    def prune(self):
        if self.arbitor is not None:
            return
        num_diff = 0
        majory_1 = 1 if self.majarity_1() else 0
        num_diff_maj = 0
        for i in range(len(self.valid_data)):
            pred = self.predict(self.data.tdata[self.valid_data[i]])
            if pred != self.data.tlabel[self.valid_data[i]]:
                num_diff += 1
            if self.data.tlabel[self.valid_data[i]] != majory_1:
                num_diff_maj += 1

        print 'diff', num_diff, num_diff_maj, 'in', len(self.data.tdata)
        if num_diff > num_diff_maj:
            self.arbitor = majory_1 == 1


    def fit(self):
        if len(self.valid_data) == 0:
            return

        assert len(self.valid_data) != 0

        is_all_1, is_all_0 = self.check()
        if is_all_0:
            self.arbitor = False
            return
        if is_all_1:
            self.arbitor = True
            return

        if len(self.valid_features) == 0:
            self.arbitor = self.majarity_1()
            return

        self.get_feature()

        self.prune()


class DecisionTree:
    def __init__(self):
        pass

    def fit(self, training_data, training_label, continous):
        data = Data(training_data, training_label, continous)

        valid_data = [x for x in range(len(training_data))]
        valid_feature = [x for x in range(len(training_data[0]))]

        self.root = Tree(data, valid_data, valid_feature)
        self.root.fit()

    def predict(self, test_data):
        test_label = []
        for d in test_data:
            test_label.append(self.root.predict(d))
        return test_label