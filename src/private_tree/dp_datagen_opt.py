import numpy as np
import pandas as pd
from gurobi import *


###########################
# Generate Dataset from partition
###########################
def generate_unlabeled_data(partition, data, x, y, counts, bin_edges=None, seed=1234):
    rand = np.random.RandomState(seed=seed)
    p_data = []
    # Create a data partition
    for i, pslice in enumerate(partition):

        _d_gen = {v: [] for v in [y] + x}
        rows_to_generate = np.sum(counts[i])    # counts[i] is just an integer here
        if rows_to_generate == 0: continue
        for xfeat, (l, u) in pslice['bounds'].items():
            xtype = type(data[xfeat].values[0])
            _d_gen[xfeat] += list(rand.uniform(low=l, high=u, size=rows_to_generate))
            if xtype in [np.int64, np.int, np.int32, int]:
                _d_gen[xfeat] = np.round(_d_gen[xfeat]).astype(xtype)

        _d_gen[y] += [0] * counts[i]
        p_data.append(pd.DataFrame(_d_gen))

    d_gen = pd.concat([d for d in p_data], ignore_index=True)
    return d_gen.reset_index(drop=True)

def generate_levels(tree, data, x, max_lev=None):
    data_cols = list(data.columns)
    bounds = {_x: ((data[_x].min(), data[_x].max())) for _x in x}
    if max_lev is None:
        max_lev = tree._max_depth
    save_lev = { i : {'splits': [], 'counts': [], 'parent_idx': []} for i in range(max_lev+1)}

    def recur(node, lev):
        if node._split_value_from_parent is None:  # Root node
            save_lev[lev] = {'splits': {None: None}, 'counts': [node._class_counts], 'parent_idx': [None]}

            # Recur into children
            for c in node._children:
                recur(c, lev+1)

        else:  # Internal Node
            if node._split_value_from_parent.startswith('<'):
                code = '<'
            else:
                code = '>='
            split = node._svfp_numer
            parent_feat = data_cols[node._parent_node._splitting_attribute]

            lb, ub = bounds[parent_feat]
            # update LB, UB
            bounds[parent_feat] = (lb if code == '<' else split, split if code == '<' else ub)

            save_lev[lev]['splits'].append( (parent_feat, bounds[parent_feat]) )
            save_lev[lev]['counts'].append( node._class_counts )
            save_lev[lev]['parent_idx'].append(len(save_lev[lev-1]['splits'])-1)

            if lev >= max_lev or node._splitting_attribute is None: #leaf
                pass
                # partition.append({'bounds': bounds.copy(), 'counts': node._class_counts})
            else:
                for c in node._children:
                    recur(c, lev+1)

            # restore LB, UB
            bounds[parent_feat] = (lb, ub)

    recur(tree._root_node, 0)
    return save_lev

def make_partition(tree_lev, data, x):
    bounds = {_x: ((data[_x].min(), data[_x].max())) for _x in x}
    leaf_lev = len(tree_lev)-1
    partition = []
    def recur(lev, p_idx):
        if lev == 0:  # Root node
            recur(1, 0)
        else:  # Internal Node
            # retrieve all indexes matching p_idx
            ch_idx = np.where(np.asarray(tree_lev[lev]['parent_idx']) == p_idx)[0]
            for i in ch_idx:
                attr, (lb, ub) = tree_lev[lev]['splits'][i]
                ct = tree_lev[lev]['counts'][i]

                _lb, _ub = bounds[attr]  # previous bounds
                bounds[attr] = (lb, ub)  # update bounds
                if lev == leaf_lev:
                    partition.append({'bounds': bounds.copy(), 'counts': ct})
                else:
                    recur(lev+1, i)
                # restore bounds
                bounds[attr] = (_lb, _ub)

    recur(0, 0)
    return partition

def find_idx_of_ancestor(tree_lev, leaf_idx, target_lev):
    curr_lev = len(tree_lev)-1
    curr_idx = leaf_idx
    while curr_lev > target_lev:
        curr_idx = tree_lev[curr_lev]['parent_idx'][curr_idx]
        curr_lev -= 1
    return curr_idx

def optimize_counts(tree_lev, max_depth, eps, seed=1234):
    '''
        - maxdept = maximum depth onto which enforce consistency (exclusive) and including leaves
        - tree_depth =
        - seed
        - eps
    '''

    eps_p = (eps / (max_depth - 1))
    tree_depth = len(tree_lev) - 1
    nleaves = len(tree_lev[tree_depth]['parent_idx'])
    rand = np.random.RandomState(seed)
    assert (max_depth < tree_depth)

    map_leaf_idx_to_ancestor = []
    for i in range(nleaves):
        map_leaf_idx_to_ancestor.append(find_idx_of_ancestor(tree_lev, i, max_depth - 1))

    VTYPE = GRB.CONTINUOUS
    model = Model('optimize')

    """ Variables """
    vars = {}
    for lev in range(max_depth):
        n = len(tree_lev[lev]['parent_idx'])
        vars[lev] = model.addVars(list(range(n)), lb=0, vtype=VTYPE)
    # last level
    vars[max_depth] = model.addVars(list(range(nleaves)), lb=0, vtype=VTYPE)

    """ Values """
    vals = {}
    for lev in range(max_depth):
        n = len(tree_lev[lev]['parent_idx'])
        vals[lev] = [np.sum(list(tree_lev[lev]['counts'][i].values())) for i in range(n)]
    # last level
    vals[max_depth] = [np.sum(list(tree_lev[tree_depth]['counts'][i].values())) for i in range(nleaves)]

    """ Constraints """
    for lev in range(max_depth - 1):
        n = len(tree_lev[lev]['parent_idx'])
        for pidx in range(n):
            ch_idx = np.where(np.asarray(tree_lev[lev + 1]['parent_idx']) == pidx)[0]
            model.addConstr(vars[lev][pidx] == quicksum(vars[lev + 1][i] for i in ch_idx))
            # pune noisy branches
            if vals[lev][pidx] == 0:
                model.addConstr(vars[lev][pidx] == 0)

    # Last level:
    n = len(tree_lev[max_depth - 1]['parent_idx'])
    for pidx in range(n):
        ch_idx = np.where(np.asarray(map_leaf_idx_to_ancestor) == pidx)[0]
        model.addConstr(vars[max_depth - 1][pidx] == quicksum(vars[max_depth][i] for i in ch_idx))

        # pune noisy branches
        if vals[max_depth - 1][pidx] == 0:
            model.addConstr(vars[max_depth - 1][pidx] == 0)

    for i in range(nleaves):
        if vals[max_depth][i] == 0:
            model.addConstr(vars[max_depth][i] == 0)

    """ Objective """
    obj = None
    for lev in range(max_depth):
        n = len(tree_lev[lev]['parent_idx'])
        noisy_count = vals[lev] + rand.laplace(0, 1 / eps_p, n)
        for i in range(n):
            obj += 1 / n * (vars[lev][i] - noisy_count[i]) * (vars[lev][i] - noisy_count[i])
    # Last Level
    noisy_count = vals[max_depth]  # already noisy
    for i in range(nleaves):
        obj += 1 / nleaves * (vars[max_depth][i] - noisy_count[i]) * (vars[max_depth][i] - noisy_count[i])

    """Solve"""
    model.setObjective(obj)
    model.setParam('OutputFlag', False)
    model.optimize()

    return [np.round(vars[max_depth][i].x).astype(int) for i in range(nleaves)]


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from private_tree.tree_split import DP_Random_Forest
    from data_loader import (load_dataset, allocate_data_to_agents)

    data, x, y, p, c = load_dataset('census', 'education-num')
    data = data[0:10]

    mdl = DP_Random_Forest(train=data.values, categs=[], num_trees=1, max_tree_depth=4, seed=2)
    mdl.fit(train=data.values, eps=1.0)
    mdl.predict(test=data.values)
    # Agent Generate private (unlabeled) data


    tree_lev  = generate_levels(mdl._trees[0], data, x)
    partition = make_partition(tree_lev, data, x)
    counts = optimize_counts(tree_lev, 3, 1.0, 1)
    d = generate_unlabeled_data(partition, data, x, y, counts)
    print(d)
