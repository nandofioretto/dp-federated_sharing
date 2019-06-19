import numpy as np

class Partitioner:
    def __init__(self, data, X_feat, y_feat, n_classes_X=None, n_classes_y=None, seed=1234):
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
        self.X_feat = X_feat
        self.y_feat = y_feat
        self.X = data[X_feat]
        self.y = data[[y_feat]]
        self.rand = np.random.RandomState(seed)

        self.n_classes_X = n_classes_X
        self.n_classes_y = n_classes_y
        self.y_classes = None
        self.X_classes = None
        self.partition = None
