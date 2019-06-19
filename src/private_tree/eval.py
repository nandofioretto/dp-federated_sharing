''' A toy example of how to call the class '''

from private_tree.tree import DP_Random_Forest
import numpy as np
from data_loader import load_dataset

###########################
# Generate Dataset from partition
###########################
def generate_data(partition, x, y, bin_edges=None, eps=None, seed=1234):
    import pandas as pd
    d_gen = pd.DataFrame()
    rand = np.random.RandomState(seed=seed)
    # Create a data partition
    for pslice in partition:
        ycounts = np.asarray(list(pslice['counts'].values()))
        if eps is not None:
            ycounts += rand.laplace(0, 1 / eps, size=len(ycounts)).astype(int)
            ycounts[ycounts < 0] = 0
        _d_gen = {v: [] for v in x + [y]}
        rows_to_generate = np.sum(ycounts)
        if rows_to_generate == 0: continue
        for xfeat, (l, u) in pslice['bounds'].items():
            xtype = type(data[xfeat][0])
            _d_gen[xfeat] += list(rand.uniform(low=l, high=u, size=rows_to_generate))
            if xtype in [np.int64, np.int, np.int32, int]:
                _d_gen[xfeat] = np.round(_d_gen[xfeat]).astype(xtype)

        if bin_edges is not None:
            for i in range(len(ycounts)):
                l, u = bin_edges[i], bin_edges[i+1]
                _d_gen[y] += list(rand.uniform(low=l, high=u, size=ycounts[i]))
        else:
            for i in range(len(ycounts)):
                _d_gen[y] += [i] * ycounts[i]

        d_gen = d_gen.append(pd.DataFrame(_d_gen), sort=False)
    return d_gen.reset_index(drop=True)

###########################
# Generate Partition
###########################
def generate_partition(tree, data, x):
    data_cols = list(data.columns)
    bounds = {_x: ((data[_x].min(), data[_x].max())) for _x in x}
    partition = []
    def recur(node):
        if node._split_value_from_parent is None: # Root node
            # Recur into children
            for c in node._children:
                recur(c)
        else: # Internal Node
            code = '<' if node._split_value_from_parent.startswith('<') else '>='
            split = node._svfp_numer
            parent_feat = data_cols[node._parent_node._splitting_attribute]

            lb, ub = bounds[parent_feat]
            # update LB, UB
            bounds[parent_feat] = (lb if code == '<' else split, split if code == '<' else ub)

            if node._splitting_attribute is None:  # Leaf
                partition.append({'bounds': bounds.copy(), 'counts': node._class_counts})
            else:
                for c in node._children:
                    recur(c)

            # restore LB, UB
            bounds[parent_feat] = (lb, ub)

    recur(tree._root_node)
    return partition


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import KBinsDiscretizer

    data, x, y, p, c = load_dataset(name='census', pfeat='education-num', to_numeric=True)
    smalldb = data[1:10000].values
    cat = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

    ## at root node:
    print(np.bincount(smalldb[:,0], minlength=2))
    print(smalldb[smalldb[:,2] == 4])

    cat_idx = [list(data.columns).index(i) for i in cat]
    eps = 1
    # # forest = DP_Random_Forest(Dtrain.values, Dtest.values, [], 100, 1, 10)
    forest = DP_Random_Forest(smalldb, cat_idx, 10, 10)
    forest.fit(smalldb, eps)
    pred = forest.predict(data[10000:11000].values)
    print(forest._accuracy)
    # partition = generate_partition(forest._trees[0], data[1:1000], x)
    # private_data = generate_data(partition, x, y, bin_edges=enc.bin_edges_[0], eps=eps, seed=1234)
    #
    # print(private_data)

