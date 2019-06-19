from partitioner import Partitioner
from sklearn import tree
from sklearn.tree import _tree
from utils import to_multinomial
import numpy as np
import pandas as pd

class TreePartitioner(Partitioner):
    def __init__(self, data, X_feat, y_feat, hp, seed=1234):
        super(TreePartitioner, self).__init__(data, X_feat, y_feat, None, seed)
        '''
        :param X:
        :param y:
        :param feat_names: The X attributes
        :param n_yclasses: If not None, it will transform y to a multinomial variable
        :param tree_dept: Maximal tree depth
        :param test_size: test/train size split
        :param seed: the random seed to adopt
        '''
        self.data = data
        self.x_feat = X_feat
        self.y_feat = y_feat
        X = data[X_feat]
        y = data[[y_feat]]

        n_yclasses = hp['y-classes']
        self.y_classes = {}
        if n_yclasses is not None:
            y, self.y_classes = to_multinomial(y, n_yclasses)

        self.clf = tree.DecisionTreeClassifier(random_state=seed,
                                               criterion=hp['criterion'],
                                               max_depth=hp['max_depth'],
                                               min_samples_leaf=hp['min_samples_leaf'],
                                               min_samples_split=hp['min_samples_split'])
        self.clf = self.clf.fit(X, y)
        self.X_feat_dic = {v: (X[[v]].values.min(), X[[v]].values.max()) for v in X_feat}
        self.tree_splits = []
        self.partition = []
        self.rand = np.random.RandomState(seed)

    @property
    def get_tree(self):
        return self.clf

    def get_bounds(self, x, path):
        S = [(l, u) for (f, l, u) in path if f == x]
        if len(S) > 0:
            idx = np.argmin([u - l for (l, u) in S])
            return S[idx]
        else:
            return self.X_feat_dic[x]

    def make_tree_split(self):
        self.tree_splits = []
        tree_ = self.clf.tree_
        feature_name = [
            self.X_feat[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        feature_values = self.X_feat_dic.copy()
        self.tree_splits = []

        def recurse(parent_node, parent_dir, node, depth):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                (lb, ub) = feature_values[name]
                ########### Randomization ################
                threshold = np.round(tree_.threshold[node], 2) #+ self.rand.laplace(0, (ub-lb) / 10)
                if threshold > ub: threshold = ub
                if threshold < lb: threshold = lb
                ##########################################

                feature_values[name] = (lb, threshold)
                #assert feature_values[name][0] <= feature_values[name][1]

                # print('Lev-', depth, '::', name, np.round(feature_values[name], 2), ':', tree_.value[node][0].astype(int))
                self.tree_splits.append({'node': node,
                                         'dir': 'L',
                                         'pnode': parent_node,
                                         'pdir': parent_dir,
                                         'lev': depth,
                                         'feature': name,
                                         'bounds': feature_values[name]})
                recurse(node, 'L', tree_.children_left[node], depth + 1)

                feature_values[name] = (threshold, ub)
                #assert feature_values[name][0] <= feature_values[name][1]

                # print('Lev-', depth, '::', name, np.round(feature_values[name], 2), ':', tree_.value[node][0].astype(int))
                self.tree_splits.append({'node': node,
                                         'dir': 'R',
                                         'pnode': parent_node,
                                         'pdir': parent_dir,
                                         'lev': depth,
                                         'feature': name,
                                         'bounds': feature_values[name]})
                recurse(node, 'R', tree_.children_right[node], depth + 1)

                # Restore LB, UB
                feature_values[name] = (lb, ub)
                assert feature_values[name][0] <= feature_values[name][1]

            else:
                pass
                ## Generate entries for New x
                # self.tree_splits.append((feature_values,
                #                        tree_.value[node][0].astype(int))) ## these are the values of the parent! WRONG

        recurse(None, None, 0, 1)
        return self.tree_splits

    def get_partition(self):

        def push_property(pstack: list, prop: dict, data: pd.DataFrame, inline=True):
            feat, (l, u) = prop['feature'], prop['bounds']
            if inline:
                pstack.append((data[feat] >= l) & (data[feat] <= u))
                return
            else:
                return pstack + [(data[feat] >= l) & (data[feat] <= u)]

        def pop(pstack: list, ntimes=1):
            assert ntimes <= len(pstack)
            # pstack = pstack[:-ntimes]
            # return pstack
            for i in range(ntimes):
                pstack.pop()

        def count_cl_items(data: pd.DataFrame, pstack: list, clname: str, cl_size: int):
            return np.bincount(data[np.asarray(pstack).all(axis=0)][clname].values, minlength=cl_size)

        self.make_tree_split()
        n_y_classes = len(self.y_classes)
        data = self.data
        pr_lev, _stack_prop, _store_path, paths = 0, [], [], []
        for i, prop in enumerate(self.tree_splits):
            cr_lev, f, (l,u) = prop['lev'], prop['feature'], prop['bounds']

            if cr_lev == pr_lev + 1: # descend
                pass
            elif cr_lev == pr_lev: # sibling
                paths.append(_store_path.copy())
                pop(_stack_prop)
                pop(_store_path)
            elif cr_lev < pr_lev: # backtrack / backjump
                paths.append(_store_path.copy())
                dist = pr_lev - cr_lev + 1
                pop(_stack_prop, dist)
                pop(_store_path, dist)

            push_property(_stack_prop, prop, data)
            _store_path.append((f, l, u))
            #print(cr_lev, f, '(', l, u, '):', count_cl_items(data, _stack_prop, y_feat, 3))

            prop['count'] = count_cl_items(data, _stack_prop, self.y_feat, n_y_classes)

            # memorize current level info
            pr_lev = cr_lev

        ## Create partition:
        self.partition = []
        for path in paths:
            self.partition.append([(x, self.get_bounds(x, path)) for x in self.x_feat])

        return self.partition

if __name__ == '__main__':
    import pandas as pd
    from sklearn import preprocessing

    dataset_path = '/Users/fferdinando3/Repos/differential_privacy/dp-distr-ml/datasets/'
    dataset = pd.read_csv(dataset_path + 'fishiris.csv', na_values='?', skipinitialspace=True)
    lb_make = preprocessing.LabelEncoder()
    obj_df = dataset.select_dtypes(include=['object']).copy()
    for feat in list(obj_df.columns):
        dataset.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    x_feat = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
    y_feat = 'Name'
    p_feat = 'SepalLength'
    print(len(dataset))
    params_fit = {'epsilon': 1,
                  'y-classes': 3,
                  'criterion': 'gini',
                  'max_depth': 5,
                  'min_samples_leaf': 1,
                  'min_samples_split': 10}

    tree1 = TreePartitioner(dataset, x_feat, y_feat, params_fit)
    partition = tree1.get_partition()

    _parts = {x: [(dataset[x].min(), dataset[x].max())] for x in x_feat}
    #_parts = {}

    for p in partition:
        for a, (l, u) in p:
            if (l, u) in _parts[a]:
                continue
            # find interval which contains (l, u)
            for i, (_l, _u) in enumerate(_parts[a]):
                if l >= _l and u <= _u:
                    del _parts[a][i]
                    if l > _l:
                        _parts[a].append((_l, l))
                    _parts[a].append((l, u))
                    if u < _u:
                        _parts[a].append((u, _u))
                    break
            _parts[a] = sorted(_parts[a], key=lambda x: (x[0], x[1]))

    for part in _parts:
        print(part, _parts[part])


    # for split in tree1.tree_splits:
    #     print (split)

    # from preprocessing.data_generator import generate_private_data
    # d = generate_private_data(dataset, tree1, 1.0)
    # print(d)

    # from dp_data_generator import generate_opt_private_data
    # d = generate_opt_private_data(dataset, tree1, 1.0)
