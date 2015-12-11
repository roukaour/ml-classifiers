__author__ = 'Jian Yang'
__date__ = '11/21/15'

import math
import copy

pt = {0: {'a': 1, 'b': 0}, 1: {}, 2: {}, 3: {'y': 1, 'u': 0, 'l': 2}, 4: {'gg': 2, 'p': 1, 'g': 0}, 5: {'aa': 11, 'c': 6, 'e': 10, 'd': 7, 'cc': 4, 'k': 5, 'j': 13, 'm': 2, 'q': 1, 'i': 9, 'r': 3, 'ff': 12, 'w': 0, 'x': 8}, 6: {'bb': 2, 'h': 1, 'dd': 7, 'j': 4, 'o': 6, 'n': 8, 'ff': 3, 'v': 0, 'z': 5}, 7: {}, 8: {'t': 0, 'f': 1}, 9: {'t': 0, 'f': 1}, 10: {}, 11: {'t': 1, 'f': 0}, 12: {'p': 2, 's': 1, 'g': 0}, 13: {}, 14: {}}

def get_feature(idx, i):
    for k, v in pt[idx].items():
        if v == i:
            return k
    assert(0)

def getlog(num, total):
    ratio = num * 1.0 / total
    if ratio == 0:
        return 0
    assert ratio >= 0
    return ratio * math.log(ratio, 2)

def get_entropy(a0, a1, a2, a3):
    assert a0+a1+a2+a3 != 0
    s1 = a0+a1
    s2 = a2+a3
    s = s1+s2
    ta1 = -getlog(a0, s1) -getlog(a1,s1)
    ta2 = -getlog(a2, s2) -getlog(a3,s2)
    n0 = a0+a2
    n1 = a1+a3
    return -getlog(n0, n0+n1)-getlog(n1, n0+n1)-(ta1 * s1*1.0 / s + ta2 * s2*1.0 / s)

def split(alist):
    res = []
    for i in range((2 ** (len(alist)-1))):
        v1 = [alist[0]]
        v2 = []
        for k in range(len(alist)-1):
            if i & (1 << k):
                v1.append(alist[k+1])
            else:
                v2.append(alist[k+1])
        if (len(v1) == 0 or len(v2) == 0):
            continue
        res.append((set(v1), set(v2)))
    return res

class Data:
    def __init__(self, tdata, tlabel, continous):
        self.tdata = tdata
        self.tlabel = tlabel
        self.continous = continous

class TreeNode:
    def __init__(self):
        self.arb_pred = None
        self.arb_tt = None

        self.feat_idx = None
        self.feat_cont = None
        self.feat_value = None

        self.trees = {}

    def pdot(self, dotf, num = 0):
        mynum = num
        if self.arb_pred is not None:
            print >> dotf, "%d [label=\"%d\", shape=\"box\"] ;" % (mynum, self.arb_pred)
            return mynum + 1

        if self.feat_cont:
            print >> dotf, "%d [label=\"X[%d] <= %.2f\", shape=\"box\"] ;" % (mynum, self.feat_idx+1, self.feat_value)
            print >> dotf, "%d -> %d [ label = \"Yes\" ];" % (mynum, mynum+1)
            num_l = self.trees[0].pdot(dotf, mynum+1)
            print >> dotf, "%d -> %d [ label = \"No\" ];" % (mynum, num_l)
            num_r = self.trees[1].pdot(dotf, num_l)
            return num_r

        else:
            print >> dotf, "%d [label=\"X[%d]\", shape=\"box\"] ;" % (mynum, self.feat_idx+1)
            next_num = mynum+1
            v1, v2 = self.feat_value

            print >> dotf, "%d -> %d [ label = \"%s\" ];" % (mynum, mynum+1, ','.join([get_feature(self.feat_idx, x)for x in v1]))
            num_l = self.trees[0].pdot(dotf, mynum+1)
            print >> dotf, "%d -> %d [ label = \"%s\" ];" % (mynum, num_l, ','.join([get_feature(self.feat_idx, x)for x in v2]))
            num_r = self.trees[1].pdot(dotf, num_l)
            return num_r

    def size(self):
        s = 1
        for t in self.trees.values():
            s += t.size()
        return s

    def predict(self, sdata):
        if self.arb_pred is not None:
            assert self.arb_pred == 0 or self.arb_pred == 1
            return self.arb_pred

        feat_value = sdata[self.feat_idx]

        if self.feat_cont:
            if feat_value <= self.feat_value:
                res = self.trees[0].predict(sdata)
            else:
                res = self.trees[1].predict(sdata)
            return res
        else:
            v1, v2 = self.feat_value

            if feat_value in v1:
                res = self.trees[0].predict(sdata)
                return res
            elif feat_value in v2:
                res = self.trees[1].predict(sdata)
                return res
            else:
                return self.arb_tt

    def p(self, indent = 0):
        if self.arb_pred is not None:
            print ' ' * indent * 2, self.arb_pred
        else:
            print ' ' * indent * 2, self.feat_idx, self.feat_value, self.arb_pred
            for k, t in self.trees.items():
                print ' ' * (indent+1) * 2, k, "->"
                t.p(indent+1)

class Tree:
    def __init__(self, data, valid_data, valid_feature, parent_depth):
        self.data = data
        self.valid_data = valid_data
        self.valid_features = valid_feature
        self.num_data = len(self.valid_data)

        self.root = TreeNode()
        self.trees = {}
        self.depth = parent_depth+1

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
        assert num_0 + num_1 == len(self.valid_data)
        res = 1 if num_1 >= num_0 else 0
        return res

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
            assert num_l0+num_g0 != 0
            assert num_l1+num_g1 != 0

            sub_entropy = get_entropy(num_l0, num_l1, num_g0, num_g1)
            if False:
                sub_entropy = (-getlog(num_l0, num_le)-getlog(num_l1, num_le)) * (num_le * 1.0 / (num_le + num_gt)) + \
                              (-getlog(num_g0, num_gt)-getlog(num_g1, num_gt)) * (num_gt * 1.0 / (num_le + num_gt))
                assert num_le + num_gt == len(self.valid_data)
                sub_entropy = -getlog(num_l0+num_g0, len(self.valid_data))-getlog(num_l1+num_g1, len(self.valid_data)) - sub_entropy

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

        num_0 = 0
        num_1 = 0
        for d in self.valid_data:
            if self.data.tlabel[d] == 1:
                num_1 += 1
            else:
                num_0 += 1
        keys = list(values.keys())
        value_splits = split(keys)
        if len(value_splits) == 0:
            return -100, None
        assert(len(value_splits) > 0)
	max_gain = -999
        max_v = None

        for v in value_splits:
            v1, v2 = v
            assert isinstance(v1, set)
            assert isinstance(v2, set)

            num_1_v1 = 0
            num_0_v1 = 0
            num_1_v2 = 0
            num_0_v2 = 0

            for d in self.valid_data:
                if self.data.tdata[d][f] in v1:
                    if self.data.tlabel[d] == 1:
                        num_1_v1 += 1
                    else:
                        num_0_v1 += 1
                else:
                    if self.data.tlabel[d] == 1:
                        num_1_v2 += 1
                    else:
                        num_0_v2 += 1
            assert num_0_v1 + num_0_v2 == num_0
            assert num_1_v1 + num_1_v2 == num_1
            entropy = get_entropy(num_0_v1, num_1_v1, num_0_v2, num_1_v2)
            if entropy > max_gain:
                max_gain = entropy
                max_v = v1, v2

        assert max_v is not None
        return max_gain, max_v

    def split_continous(self):
        f = self.root.feat_idx

        le_data = []
        gt_data = []

        le_value = set()
        gt_value = set()

        for d in self.valid_data:
            if self.data.tdata[d][f] <= self.root.feat_value:
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

        if len(le_data) != 0 and len(gt_data) != 0:
            self.trees[0] = Tree(self.data, le_data, le_features, self.depth)
            self.trees[0].build()
            assert isinstance(self.trees[0].root, TreeNode)
            self.root.trees[0] = self.trees[0].root

            self.trees[1] = Tree(self.data, gt_data, gt_features, self.depth)
            self.trees[1].build()
            assert isinstance(self.trees[1].root, TreeNode)
            self.root.trees[1] = self.trees[1].root

        else:
            self.root.arb_pred = self.majarity_1()

    def split_discrete(self):
        f = self.root.feat_idx
        v1, v2 = self.root.feat_value
        le_data = []
        gt_data = []

        for d in self.valid_data:
            if self.data.tdata[d][f] in v1:
                le_data.append(d)
            else:
                gt_data.append(d)

        gt_features = self.valid_features
        le_features = self.valid_features
        if len(v2) == 1:
            gt_features = copy.copy(self.valid_features)
            gt_features.remove(f)

        if len(v1) == 1:
            le_features = copy.copy(self.valid_features)
            le_features.remove(f)

        assert len(le_data) + len(gt_data) == len(self.valid_data)

        if len(le_data) != 0 and len(gt_data) != 0:
            self.trees[0] = Tree(self.data, le_data, le_features, self.depth)
            self.trees[0].build()
            assert isinstance(self.trees[0].root, TreeNode)
            self.root.trees[0] = self.trees[0].root

            self.trees[1] = Tree(self.data, gt_data, gt_features, self.depth)
            self.trees[1].build()
            assert isinstance(self.trees[1].root, TreeNode)
            self.root.trees[1] = self.trees[1].root

        else:
            self.root.arb_pred = self.majarity_1()

    def split(self, feat_idx, split_value): # split_value is for countinous feature only
        self.root.feat_idx = feat_idx
        self.root.feat_value = split_value

        if self.data.continous[feat_idx]:
            self.root.feat_cont = True
            self.split_continous()
        else:
            self.root.feat_cont = False
            self.split_discrete()

    def build_subtrees(self):
        entropy_feature = {}
        cont_feature = {}
        for f in self.valid_features:
            assert f < len(self.data.continous), f
            if self.data.continous[f]: # continous
                max_entropy, max_value = self.get_entropy_countinues(f)
                entropy_feature[f] = max_entropy
                cont_feature[f] = max_value
            else:
                entropy_feature[f], max_v = self.get_entropy_discrete(f)
                cont_feature[f] = max_v

        max_e = max(entropy_feature.values())
        for f, v in entropy_feature.items():
            if v == max_e:
                self.split(f, cont_feature[f] if cont_feature.has_key(f) else None)
                break

    def build(self):
        assert len(self.valid_data) != 0

        is_all_1, is_all_0 = self.check()
        if is_all_0:
            self.root.arb_pred = 0
            return
        if is_all_1:
            self.root.arb_pred = 1
            return
        if len(self.valid_features) == 0 or self.depth > 100:
            self.root.arb_pred = self.majarity_1()
            return

        self.root.arb_tt = self.majarity_1()
        self.build_subtrees()

    def prune(self, verify_data):
        if self.root.arb_pred is not None:
            return

        num_diff = 0
        majory_1 = self.majarity_1()
        num_diff_maj = 0
        for v in verify_data:
            pred = self.root.predict(self.data.tdata[v])
            if pred != self.data.tlabel[v]:
                num_diff += 1
            if self.data.tlabel[v] != majory_1:
                num_diff_maj += 1

        if num_diff > num_diff_maj * 1.2:
            self.root.arb_pred = majory_1
        else:
            if self.root.feat_cont:
                valid_le = [x for x in verify_data if self.data.tdata[x][self.root.feat_idx] <= self.root.feat_value]
                valid_ge = [x for x in verify_data if self.data.tdata[x][self.root.feat_idx] > self.root.feat_value]
                assert len(valid_le) + len(valid_ge) == len(verify_data)
                self.trees[0].prune(valid_le)
                self.trees[1].prune(valid_le)
            else:
                for v, t in self.trees.items():
                    valid_sub = [x for x in verify_data if self.data.tdata[x][self.root.feat_idx] == v]
                    t.prune(valid_sub)


class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, training_data, training_label, continous):
        data = Data(training_data, training_label, continous)

        valid_data = [x for x in range(len(training_data))]
        valid_feature = [x for x in range(len(training_data[0]))]

        do_verify = True
        num_verify = len(training_label) / 2
        if do_verify:
            verify_data = valid_data[num_verify:]
            valid_data = valid_data[:num_verify]

        tree_builder = Tree(data, valid_data, valid_feature, 0)
        tree_builder.build()

        self.root = tree_builder.root
        assert(self.root)
        self.dotp('1.dot')

        if do_verify:
            tree_builder.prune(verify_data)

        self.root = tree_builder.root
        self.dotp('2.dot')

    def predict(self, test_data):
        test_label = []
        for d in test_data:
            pred = self.root.predict(d)
            assert not isinstance(pred, bool)
            test_label.append(pred)
        return test_label

    def dotp(self, fn):
        with open(fn, 'w') as dotf:
            print >> dotf, "digraph Tree {"
            self.root.pdot(dotf)
            print >> dotf, "}"
