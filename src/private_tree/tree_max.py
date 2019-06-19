''' A class that builds Differentially Private Random Decision Trees, using Smooth Sensitivity

    Code modified by Sam Fletcher
'''
from collections import Counter, defaultdict
import random
import numpy as np
import math
import multiprocessing as multi

MULTI_THREAD = False  # Turn this on if you would like to use multi-threading. Warning: if each tree builds too quickly, overhead time will be relatively large.


def expo_mech(e, s, counts):
    ''' For this implementation of the Exponetial Mechanism, we use a piecewise linear scoring function,
    where the element with the maximum count has a score of 1, and all other elements have a score of 0. '''
    weighted = []
    max_count = max([v for k, v in counts.items()])

    for label, count in counts.items():
        ''' if the score is non-monotonic, s needs to be multiplied by 2 '''
        if count == max_count:
            if s < 1.0e-10:
                power = 50  # e^50 is already astronomical. sizes beyond that dont matter
            else:
                power = min(50, (e * 1) / (2 * s))  # score = 1
        else:
            power = 0  # score = 0
        weighted.append([label, math.exp(power)])
    sum = 0.
    for label, count in weighted:
        sum += count
    for i in range(len(weighted)):
        weighted[i][1] /= sum
    customDist = stats.rv_discrete(name='customDist',
                                   values=([lab for lab, cou in weighted], [cou for lab, cou in weighted]))
    best = customDist.rvs()
    # print("best_att examples = "+str(customDist.rvs(size=20)))
    return int(best)

def get_partition(data, i, lb, ub):
    return data[(data[:, i] > lb) & (data[:, i] <= ub)]


class DP_Random_Forest_Max:
    ''' Make a forest of Random Trees, then filter the training data through each tree to fill the leafs.
    IMPORTANT: the first attribute of both the training and testing data MUST be the class attribute. '''

    def __init__(self, train, max_tree_depth=10, depth_non_random_tree=3, nbins=10, categs=[], num_trees=1,
                 seed=1234, keep_trees=True):
        ''' Some initialization '''
        self._categs = categs
        numers = [x + 1 for x in range(len(train[0]) - 1) if
                  x + 1 not in categs]  # the indexes of the numerical (i.e. continuous) attributes
        self._attribute_domains = self.get_attr_domains(train, numers, categs)
        self._max_depth = self.calc_tree_depth(len(numers), len(categs), max_tree_depth)
        self._num_trees = num_trees
        self._initial_num_trees = num_trees
        ''' Some bonus information gained throughout the algorithm '''
        self._missed_records = []
        self._flipped_majorities = []
        self._av_sensitivity = []
        self._empty_leafs = []
        self.seed = seed
        self.rand = np.random.RandomState(seed)
        self.keep_trees = keep_trees
        self._trees = []

        assert(depth_non_random_tree <= max_tree_depth)
        self._nrand_depth = depth_non_random_tree

        self._curr_data = None
        self._curr_bins = {}
        self._score = {}

        for i, vals in self._attribute_domains.items():
            si = set(train[:,i])
            if len(si) <= nbins:
                si = sorted(list(si))
                self._curr_bins[i] = [[si[j], si[j+1]] for j in range(len(si)-1)]
            else:
                _tmp = np.linspace(vals[0], vals[1], nbins + 1)
                self._curr_bins[i] = [[_tmp[i], _tmp[i+1]] for i in range(nbins)]
            self._curr_bins[i][0][0] -= 0.000001

        #random.shuffle(train)
        self.class_labels = list(set([int(x[0]) for x in train]))  # domain of labels

    def fit(self,train, eps):
        attribute_indexes = [k for k, v in self._attribute_domains.items()]

        subset_size = int(len(train) / self._num_trees)  # by using subsets, we don't need to divide the epsilon budget
        for i in range(self._num_trees):
            tree = self.build_tree(attribute_indexes, train[i * subset_size:(i + 1) * subset_size], eps,
                                   self.class_labels)
            if self.keep_trees:
                self._trees.append(tree)

    # Remove all trees except its onw
    def reset_trees(self):
        self._trees = self._trees[:self._initial_num_trees]
        self._num_trees = self._initial_num_trees

    def add_trees(self, trees, reset=False):
        if reset:
            self.reset_trees()
        self._trees += trees
        self._num_trees += len(trees)

    def predict(self, test):
        actual_labels = [int(x[0]) for x in test]  # ordered list of the test data labels
        voted_labels = [defaultdict(int) for x in test]  # initialize

        for tree in self._trees:
            results = self.evaluate_test(tree, test)
            ''' Collect the predictions and the bonus information '''
            curr_votes = results['voted_labels']
            for rec_index in range(len(test)):
                for lab in self.class_labels:
                    voted_labels[rec_index][lab] += curr_votes[rec_index][lab]
            self._missed_records.append(results['missed_records'])
            self._flipped_majorities.append(results['flipped_majorities'])
            self._av_sensitivity.append(results['av_sensitivity'])
            self._empty_leafs.append(results['empty_leafs'])

        final_predictions = []
        for i, rec in enumerate(test):
            final_predictions.append(Counter(voted_labels[i]).most_common(1)[0][0])
        # print(final_predictions)
        # print(actual_labels)
        counts = Counter([x == y for x, y in zip(final_predictions, actual_labels)])
        self._predicted_labels = final_predictions
        self._accuracy = float(counts[True]) / len(test)
        return final_predictions

    def get_attr_domains(self, data, numers, categs):
        attr_domains = {}
        transData = np.transpose(data)
        for i in categs:
            attr_domains[i] = [x for x in set(transData[i])]
            print("original domain length of categ att {}: {}".format(i, len(attr_domains[i])))
        for i in numers:
            vals = [float(x) for x in transData[i]]
            attr_domains[i] = [min(vals), max(vals)]
        return attr_domains

    def calc_tree_depth(self, num_numers, num_categs, max_depth):
        if num_numers < 1:  # if no numerical attributes
            return math.floor(num_categs / 2.)  # depth = half the number of categorical attributes
        else:
            ''' Designed using balls-in-bins probability. See the paper for details. '''
            m = float(num_numers)
            depth = 0
            expected_empty = m  # the number of unique attributes not selected so far
            while expected_empty > m / 2.:  # repeat until we have less than half the attributes being empty
                expected_empty = m * ((m - 1.) / m) ** depth
                depth += 1
            final_depth = math.floor(depth + (
                        num_categs / 2.))  # the above was only for half the numerical attributes. now add half the categorical attributes
            ''' WARNING: The depth translates to an exponential increase in memory usage. Do not go above ~15 unless you have 50+ GB of RAM. '''
            return min(max_depth, final_depth)

    def build_tree(self, attribute_indexes, train, epsilon, class_labels, test):
        root = self.rand.choice(attribute_indexes)
        tree = Tree(attribute_indexes, root, self, self.seed)
        if self.keep_trees:
            self._trees.append(tree)

        tree.filter_training_data_and_count(train, epsilon, class_labels)
        missed_records = tree._missed_records
        flipped_majorities = tree._flip_fraction
        av_sensitivity = tree._av_sensitivity
        empty_leafs = tree._empty_leafs
        voted_labels = [defaultdict(int) for x in test]
        for i, rec in enumerate(test):
            label = tree.classify(tree._root_node, rec)
            voted_labels[i][label] += 1
        del tree
        return {'voted_labels': voted_labels, 'missed_records': missed_records,
                'flipped_majorities': flipped_majorities,
                'av_sensitivity': av_sensitivity, 'empty_leafs': empty_leafs}


    def max_score(self, i, data, bins):
        ## sens = 1
        ncls = len(self.class_labels)
        score = 0
        for (l, u) in bins:
            d = get_partition(data, i, l, u)
            score += max(np.bincount(d[:, 0], minlength=ncls))
        return score

    def gini_score(self, i, data, bins):
        ## sens = 2
        ncls = len(self.class_labels)
        score = 0
        for (l, u) in bins:
            d = get_partition(data, i, l, u)
            size_d = len(d)
            if size_d == 0: continue
            cl = np.bincount(d[:, 0], minlength=ncls)
            score += size_d * (1 - np.sum( (cl/size_d)**2))
        return -score

    def select_split(self, attribute_domains, attr_idx):
        if self._dp_method == 'rand':
            split_val = self.rand.uniform(attribute_domains[attr_idx][0], attribute_domains[attr_idx][1])
        else:
            bins = self._curr_bins[attr_idx]
            splits, score = [], {}
            for i, (l, u) in enumerate(bins):
                split = self.rand.uniform(low=l, high=u)
                _bins = [(l, split), (split, u)]
                if self._dp_method == 'max':
                    score[i] = self.max_score(i, self._curr_data, _bins)
                elif self._dp_method == 'gini':
                    score[i] = self.gini_score(i, self._curr_data, _bins)
                splits.append(split)
            split_val = splits[expo_mech(self.eps, 1, score)]
        return split_val


    def build_tree(self, attribute_indexes, train, epsilon, class_labels):
        self._curr_data = train.copy()

        if self._dp_method == 'max':
            self._score = {i : self.max_score(i, self._curr_data, self._curr_bins) for i in self._curr_bins}
            root = expo_mech(epsilon, 1, self._score)
        elif self._dp_method == 'gini':
            self._score = {i : self.gini_score(i, self._curr_data, self._curr_bins) for i in self._curr_bins}
            root = expo_mech(epsilon, 1, self._score)
        else:
             root = self.rand.choice(attribute_indexes)

        tree = Tree(attribute_indexes, root, self, self.seed)
        tree.filter_training_data_and_count(train, epsilon, class_labels)
        return tree

    def evaluate_test(self, tree, test):
        missed_records = tree._missed_records
        flipped_majorities = tree._flip_fraction
        av_sensitivity = tree._av_sensitivity
        empty_leafs = tree._empty_leafs
        voted_labels = [defaultdict(int) for x in test]
        for i, rec in enumerate(test):
            label = tree.classify(tree._root_node, rec)
            voted_labels[i][label] += 1
        if not self.keep_trees:
            del tree
        return {'voted_labels': voted_labels, 'missed_records': missed_records,
                'flipped_majorities': flipped_majorities,
                'av_sensitivity': av_sensitivity, 'empty_leafs': empty_leafs}

class Tree(DP_Random_Forest_Max):
    ''' Set the root for this tree and then start the random-tree-building process. '''

    def __init__(self, attribute_indexes, root_attribute, pc, seed=1234):
        self._id = 0
        self._categs = pc._categs
        self._max_depth = pc._max_depth
        self._num_leafs = 0
        self.seed = seed
        self.rand = np.random.RandomState(seed)

        root = node(None, None, root_attribute, 1, 0, [], seed=self.seed)  # the root node is level 1
        attribute_domains = pc._attribute_domains

        if root_attribute not in self._categs:  # numerical attribute
            split_val = self.select_split(attribute_domains, root_attribute)


            left_domain = {k: v if k != root_attribute else [v[0], split_val] for k, v in
                           attribute_domains.items()}
            right_domain = {k: v if k != root_attribute else [split_val, v[1]] for k, v in
                            attribute_domains.items()}
            root.add_child(self.make_children([x for x in attribute_indexes], root, 2, '<' + str(split_val), split_val,
                                              left_domain))  # left child
            root.add_child(self.make_children([x for x in attribute_indexes], root, 2, '>=' + str(split_val), split_val,
                                              right_domain))  # right child
        else:  # categorical attribute
            assert(True)
            for value in attribute_domains[root_attribute]:
                root.add_child(
                    self.make_children([x for x in attribute_indexes if x != root_attribute], root, 2, value, None,
                                       attribute_domains))  # categorical attributes can't be tested again
        self._root_node = root

    ###############################
    # Take in input a specific hight up to which construct the tree with (max sensitivity)
    # 1. construct active bins [l , u] * BINS for continuous categories or [(l, u) .... (l, u)] for other cat
    # 2. at root:
    #    compute noisy splits for each attribute
    #    select attribute based on noisy split value function (MAX)
    # 3. Make split, update dataset Left  --> curr_data = curr_data[curr_data[:,attr_idx] < split]
    #                               Right --> curr_data = curr_data[curr_data[:,attr_idx] >= split]
    #  continue ...

    def select_noisy_split(self, attr, attr_bins, eps):
        # attr_bins = is a list of = [(l, u), (l, u)]
        a_lb, a_ub = self.current_attr_domains[attr]
        active_bins = [(l, u) for (l, u) in attr_bins if l >= a_lb and u <= a_ub]
        for (l, u) in active_bins:
            # count the number of items in each class:
            ch_data = data[(data[:,attr_idx] > l) & (data[:,attr_idx] <= u)]

            np.bitcount(ch_data)



    ''' Recursively make all the child nodes for the current node, until a termination condition is met. '''

    def make_children(self, candidate_atts, parent_node, current_depth, splitting_value_from_parent, svfp_numer,
                      attribute_domains):
        self._id += 1
        if not candidate_atts or current_depth >= self._max_depth + 1:  # termination conditions. leaf nodes don't count to the depth.
            self._num_leafs += 1
            return node(parent_node, splitting_value_from_parent, None, current_depth, self._id, None,
                             svfp_numer=svfp_numer, seed=self.seed)
        else:

            ##########################################
            ## todo: Change the splitting attribute!
            if self._dp_method == 'rand':
                new_splitting_attr = self.rand.choice(candidate_atts)  # pick the attribute that this node will split on
            elif self._dp_method == 'max':
                # update curr data and curr bins
                data_lev = self.curr_data.copy()        ## very memory inefficient
                bins_lev = self._curr_bins.copy()
                p_attr_idx = parent_node._splitting_attribute
                p_split = splitting_value_from_parent

                bins_l, bins_r = [], []
                i = 0
                while True:
                    l, u = self._curr_bins[p_attr_idx][i]
                    if u < p_split: bins_l.append([l, u])
                    else: bins_l.append([l, p_split]); break
                    i += 1
                bins_r.append(p_split, u)
                bins_r += [self._curr_bins[p_attr_idx][j] for j in range(i+1, len(self._curr_bins[p_attr_idx][i]))]

                # Left
                self._curr_bins[p_attr_idx] = bins_l
                self.curr_data = get_partition(p_attr_idx, bins_l[0][0], bins_l[-1][1])
                # todo: continue from here
                split_val = self.select_split(attribute_domains, p_attr_idx)

                # Right
                self._curr_bins[p_attr_idx] = bins_r
                self.curr_data = get_partition(p_attr_idx, bins_r[0][0], bins_r[-1][1])


            elif self._dp_method == 'gini':
                pass

            ##########################################
            current_node = node(parent_node, splitting_value_from_parent, new_splitting_attr, current_depth,
                                     self._id, [], svfp_numer=svfp_numer,seed=self.seed)  # make a new node

            if new_splitting_attr not in self._categs:  # numerical attribute
                ##########################################
                ## todo: Change the splitting value
                ##########################################
                split_val = self.rand.uniform(attribute_domains[new_splitting_attr][0],
                                              attribute_domains[new_splitting_attr][1])
                left_domain = {k: v if k != new_splitting_attr else [v[0], split_val] for k, v in
                               attribute_domains.items()}
                right_domain = {k: v if k != new_splitting_attr else [split_val, v[1]] for k, v in
                                attribute_domains.items()}
                current_node.add_child(
                    self.make_children([x for x in candidate_atts], current_node, current_depth + 1, '<', split_val,
                                       left_domain))  # left child
                current_node.add_child(
                    self.make_children([x for x in candidate_atts], current_node, current_depth + 1, '>=', split_val,
                                       right_domain))  # right child
            else:  # categorical attribute
                for value in attribute_domains[new_splitting_attr]:  # for every value in the splitting attribute
                    child_node = self.make_children([x for x in candidate_atts if x != new_splitting_attr],
                                                    current_node, current_depth + 1, value, None, attribute_domains)
                    current_node.add_child(child_node)  # add children to the new node
            return current_node

    ''' Record which leaf each training record belongs to, and then set the (noisy) majority label. '''

    def filter_training_data_and_count(self, records, epsilon, class_values):
        ''' epsilon = the epsilon budget for this tree (each leaf is disjoint, so the budget can be re-used). '''
        num_unclassified = 0.
        for rec in records:
            num_unclassified += self.filter_record(rec, self._root_node, class_index=0)
        self._missed_records = num_unclassified
        flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(epsilon, self._root_node,
                                                                                       class_values, 0, 0, [])
        self._av_sensitivity = np.mean(sensitivities)  # excludes empty leafs

        if self._num_leafs == 0:
            print("\n\n~~~ WARNING: NO LEAFS. num_unclassified = " + str(num_unclassified) + " ~~~\n\n")
            self._empty_leafs = -1.0
        else:
            self._empty_leafs = empty_leafs / float(self._num_leafs)

        if empty_leafs == self._num_leafs:
            print("\n\n~~~ WARNING: all leafs are empty. num_unclassified = " + str(num_unclassified) + " ~~~\n\n")
            self._flip_fraction = -1.0
        else:
            self._flip_fraction = flipped_majorities / float(self._num_leafs - empty_leafs)

    def filter_record(self, record, node, class_index=0):
        if not node:
            return 0.00001  # For debugging purposes. Doesn't happen in my experience
        if not node._children:  # if leaf
            node.increment_class_count(record[class_index])
            return 0.
        else:
            child = None
            if node._splitting_attribute not in self._categs:  # numerical attribute
                rec_val = record[node._splitting_attribute]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else:  # categorical attribute
                rec_val = str(record[node._splitting_attribute])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            if child is None and node._splitting_attribute in self._categs:  # if the record's value couldn't be found:
                # print(str([i._split_value_from_parent,])+" vs "+str([record[node._splitting_attribute],])+" out of "+str(len(node._children)))
                return 1.  # For debugging purposes
            elif child is None:  # if the record's value couldn't be found:
                return 0.001  # For debugging purposes
            return self.filter_record(record, child, class_index)

    def set_all_noisy_majorities(self, epsilon, node, class_values, flipped_majorities, empty_leafs, sensitivities):
        if node._children:
            for child in node._children:
                flipped_majorities, empty_leafs, sensitivities = self.set_all_noisy_majorities(
                    epsilon, child, class_values, flipped_majorities, empty_leafs, sensitivities)
        else:
            flipped_majorities += node.set_noisy_majority(epsilon, class_values)
            empty_leafs += node._empty
            if node._sensitivity >= 0.0: sensitivities.append(node._sensitivity)
        return flipped_majorities, empty_leafs, sensitivities

    def classify(self, node, record):
        if not node:
            return None
        elif not node._children:  # if leaf
            return node._noisy_majority
        else:  # if parent
            attr = node._splitting_attribute
            child = None
            if node._splitting_attribute not in self._categs:  # numerical attribute
                rec_val = record[attr]
                for i in node._children:
                    if i._split_value_from_parent.startswith('<') and rec_val < i._svfp_numer:
                        child = i
                        break
                    if i._split_value_from_parent.startswith('>=') and rec_val >= i._svfp_numer:
                        child = i
                        break
            else:  # categorical attribute
                rec_val = str(record[attr])
                for i in node._children:
                    if i._split_value_from_parent == rec_val:
                        child = i
                        break
            if child is None:  # if the record's value couldn't be found, just return the latest majority value
                return node._noisy_majority  # majority_value, majority_fraction

            return self.classify(child, record)

from scipy import stats  # for Exponential Mechanism

class node:
    def __init__(self, parent_node, split_value_from_parent, splitting_attribute, tree_level, id, children,
                 svfp_numer=None, seed=1234):
        self._parent_node = parent_node
        self._split_value_from_parent = split_value_from_parent
        self._svfp_numer = svfp_numer
        self._splitting_attribute = splitting_attribute
        # self._level = tree_level # comment out unless needed. saves memory.
        # self._id = id # comment out unless needed. saves memory.
        self._children = children
        self._class_counts = defaultdict(int)
        self._noisy_majority = None
        self._empty = 0  # 1 if leaf and has no records
        self._sensitivity = -1.0
        self.seed = seed
        self.rand = np.random.RandomState(seed)

    def add_child(self, child_node):
        self._children.append(child_node)

    def increment_class_count(self, class_value):
        self._class_counts[class_value] += 1

    def set_noisy_majority(self, epsilon, class_values):
        if not self._noisy_majority and not self._children:  # to make sure this code is only run once per leaf
            for val in class_values:
                if val not in self._class_counts: self._class_counts[val] = 0

            if max([v for k, v in self._class_counts.items()]) < 1:
                self._empty = 1
                self._noisy_majority = self.rand.choice([k for k, v in self._class_counts.items()])
                return 0  # we dont want to count purely random flips
            else:
                all_counts = sorted([v for k, v in self._class_counts.items()], reverse=True)
                count_difference = all_counts[0] - all_counts[1]
                self._sensitivity = math.exp(-1 * count_difference * epsilon)
                self._sens_of_sens = 1.
                self._noisy_sensitivity = 1.

                self._noisy_majority = self.expo_mech(epsilon, self._sensitivity, self._class_counts)

                if self._noisy_majority != int(
                        max(self._class_counts.keys(), key=(lambda key: self._class_counts[key]))):
                    # print('majority: '+str(self._noisy_majority)+' vs. max_count: '+str( max(self._class_counts.keys(), key=(lambda key: self._class_counts[key]))))
                    return 1  # we're summing the flipped majorities
                else:
                    return 0
        else:
            return 0

    def laplace(self, e, counts):
        noisy_counts = {}
        for label, count in counts.items():
            noisy_counts[label] = max(0, int(count + self.rand.laplace(scale=float(1. / e))))
        return int(max(noisy_counts.keys(), key=(lambda key: noisy_counts[key])))

    def expo_mech(self, e, s, counts):
        ''' For this implementation of the Exponetial Mechanism, we use a piecewise linear scoring function,
        where the element with the maximum count has a score of 1, and all other elements have a score of 0. '''
        weighted = []
        max_count = max([v for k, v in counts.items()])

        for label, count in counts.items():
            ''' if the score is non-monotonic, s needs to be multiplied by 2 '''
            if count == max_count:
                if s < 1.0e-10:
                    power = 50  # e^50 is already astronomical. sizes beyond that dont matter
                else:
                    power = min(50, (e * 1) / (2 * s))  # score = 1
            else:
                power = 0  # score = 0
            weighted.append([label, math.exp(power)])
        sum = 0.
        for label, count in weighted:
            sum += count
        for i in range(len(weighted)):
            weighted[i][1] /= sum
        customDist = stats.rv_discrete(name='customDist',
                                       values=([lab for lab, cou in weighted], [cou for lab, cou in weighted]))
        best = customDist.rvs()
        # print("best_att examples = "+str(customDist.rvs(size=20)))
        return int(best)

''' A toy example of how to call the class '''
if __name__ == '__main__':
    from data_loader import load_dataset
    from private_tree.tree import DP_Random_Forest

    data, x, y, p, c = load_dataset(name='census', pfeat='education-num', to_numeric=True)
    smalldb = data[1:1000].values
    cat = ['workclass', 'marital-status', 'relationship', 'race', 'sex']

    ## at root node:
    # print(np.bincount(smalldb[:, 0], minlength=2))
    # print(smalldb[smalldb[:, 2] == 4])

    cat_idx = [list(data.columns).index(i) for i in cat]
    eps = 1.0
    # forest1 = DP_Random_Forest(smalldb, [], 10, 10)
    # forest1.fit(smalldb, eps)
    # pred = forest1.predict(data[10000:11000].values)
    # print(forest1._accuracy)

    forest = DP_Random_Forest_Max(smalldb,  max_tree_depth=10, nbins=10)
    forest.fit(smalldb, eps)
    pred = forest.predict(data[10000:11000].values)
    print(forest._accuracy)
    # partition = generate_partition(forest._trees[0], data[1:1000], x)
    # private_data = generate_data(partition, x, y, bin_edges=enc.bin_edges_[0], eps=eps, seed=1234)
    #
    # print(private_data)

