from partitioner import Partitioner
from utils import to_multinomial, get_cat_indexes
import numpy as np
import itertools
from functools import reduce

class BasicPartitioner(Partitioner):
    def __init__(self, data, X_feat, y_feat, n_classes=10, seed=1234):
        '''
        :param X:
        :param y:
        :param feat_names: The X attributes
        :param n_Xclasses: The number of classes for the X partition. If the attribute is categorical then
        :param n_yclasses: If not None, it will transform y to a multinomial variable
        :param tree_dept: Maximal tree depth
        :param test_size: test/train size split
        :param seed: the random seed to adopt
        '''
        super(BasicPartitioner, self).__init__(data, X_feat, y_feat, n_classes, n_classes, seed)

    def get_partition(self):
        if self.partition is not None:
            return self.partition

        self.partition = []

        _y, self.y_classes = to_multinomial(self.y, self.n_classes_y)
        y_cat = get_cat_indexes(_y, self.y_classes)
        X_cat = {}
        self.X_classes = {}
        for f in self.X_feat:
            _x, self.X_classes[f] = to_multinomial(self.X[[f]], self.n_classes_X)
            X_cat[f] = get_cat_indexes(_x, self.X_classes[f])

        X_classes_values = list(list(self.X_classes[f].keys()) for f in self.X_feat)
        all_combos = list(itertools.product(*X_classes_values))

        for combo in all_combos:
            idx_set = reduce(np.intersect1d, (X_cat[f][combo[i]] for i, f in enumerate(self.X_feat)))
            y_cat_nrows = [len(np.intersect1d(idx_set, y_cat[i])) for i in y_cat.keys()]
            X_cat_combo = {self.X_feat[i]: self.X_classes[self.X_feat[i]][k] for i, k in enumerate(combo)}
            self.partition.append((X_cat_combo, np.asarray(y_cat_nrows)))

        return self.partition
