import numpy as np
import pandas as pd

###########################
# Generate Dataset from partition
###########################
def generate_data(partition, data, x, y, bin_edges=None, eps=None, seed=1234):
    rand = np.random.RandomState(seed=seed)
    p_data = []
    # Create a data partition
    for pslice in partition:
        ycounts = np.asarray(list(pslice['counts'].values()))
        if eps is not None:
            ycounts += rand.laplace(0, 1 / eps, size=len(ycounts)).astype(int)
            ycounts[ycounts < 0] = 0
        _d_gen = {v: [] for v in [y] + x }
        rows_to_generate = np.sum(ycounts)
        if rows_to_generate == 0: continue
        for xfeat, (l, u) in pslice['bounds'].items():
            xtype = type(data[xfeat].values[0])
            _d_gen[xfeat] += list(rand.uniform(low=l, high=u, size=rows_to_generate))
            if xtype in [np.int64, np.int, np.int32, int]:
                _d_gen[xfeat] = np.round(_d_gen[xfeat]).astype(xtype)

        # if bin_edges is not None:
        #     for i in range(len(ycounts)):
        #         l, u = bin_edges[i], bin_edges[i + 1]
        #         _d_gen[y] += list(rand.uniform(low=l, high=u, size=ycounts[i]))
        # else:
        for i in range(len(ycounts)):
            _d_gen[y] += [i] * ycounts[i]

        p_data.append(pd.DataFrame(_d_gen))

    d_gen = pd.concat([d for d in p_data], ignore_index=True)
    return d_gen.reset_index(drop=True)

###########################
# Generate Partition
# todo: Fix split with categorical attributes
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
            code = node._split_value_from_parent.split()[0]
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
    import numpy as np
    import pandas as pd
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeClassifier

    from private_tree.tree import DP_Random_Forest
    from private_tree.dp_datagen import (generate_data, generate_partition)

    from data_loader import (load_dataset, allocate_data_to_agents)
    from evaluations.eval_utils import (plot_results, save_data, eval, as_dataframe)

    data, x, y, p, c = load_dataset('census', 'education-num')
    data = data[0:10]
    ## todo: just avoid counts < 5 or so (prune branch)
    print(data)

    mdl = DP_Random_Forest(train=data.values, categs=[], num_trees=1, max_tree_depth=10, seed=1)
    mdl.fit(train=data.values, eps=1.0)
    mdl.predict(test=data.values)
    # Agent Generate private (unlabeled) data
    partition = generate_partition(mdl._trees[0], data, x)
    for p in partition:
        print(p)
    unlab_data = generate_data(partition, data, x, y, eps=1.0, seed=1)
    print(unlab_data)


