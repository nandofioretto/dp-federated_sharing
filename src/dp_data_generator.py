import numpy as np
import pandas as pd
from partitioner_tree import TreePartitioner

def generate_private_data(data:pd.DataFrame, partitioner:TreePartitioner,
                          eps:float=None, delta:float=None, seed:int=1234) -> pd.DataFrame:
    '''
        Generates a dataset using a 'paritioner'. The latter will create a partition of the attirbutes'values
        in the original datset.
        :param eps:   The privacy budget
        :param delta: The privacy failure probability
        :return: A pandas dataframe of the same type as the input dataframe
    '''
    def count_cl_items(slice, data, clname, cl_size):
        pstack = []
        for feat, (l, u) in slice:
            pstack.append((data[feat] >= l) & (data[feat] <= u))
        pstack = np.asarray(pstack)
        return np.bincount(data[pstack.all(axis=0)][clname].values, minlength=cl_size)

    rand = np.random.RandomState(seed)
    partition = partitioner.get_partition()
    distr = rand.uniform
    X_feat, y_feat, y_classes = partitioner.X_feat, partitioner.y_feat, partitioner.y_classes
    n_ycl = len(y_classes)
    eps /= 2
    d_gen = pd.DataFrame()

    for pslice in partition:
        y_counts = count_cl_items(pslice, data, y_feat, n_ycl)

        if eps is not None:
            y_counts += rand.laplace(0, 1 / eps, size=n_ycl).astype(int)
            y_counts[y_counts < 0] = 0

        _d_gen = {v: [] for v in X_feat + [y_feat]}
        # we will generate sum.counts elements
        rows_to_generate = np.sum(y_counts)

        for xfeat, (l, u) in pslice:
            xtype = type(data[xfeat][0])
            # L = self.rand.beta(0.5, 0.5, size=rows_to_generate) * (u - l) + l
            # mu, sigma = (u-l)/2, np.sqrt((u-l)/2)
            # L = self.rand.normal(mu, sigma, size=rows_to_generate)
            # L[L > u] = u
            # L[L < l] = l
            # _d_gen[key] += list(L)
            _d_gen[xfeat] += list(distr(low=l, high=u, size=rows_to_generate).astype(xtype))

        ytype = type(data[y_feat][0])
        for i, num_rows in enumerate(y_counts):
            l, u = y_classes[i]
            if num_rows > 0:
                _d_gen[y_feat] += list(distr(low=l, high=u, size=num_rows).astype(ytype))

        d_gen = d_gen.append(pd.DataFrame(_d_gen), sort=False)

    return d_gen


from gurobi import *
def generate_opt_private_data(data:pd.DataFrame,
                          partitioner:TreePartitioner,
                          eps:float=None,
                          delta:float=None,
                          max_nodes=6,
                          seed:int=1234) -> pd.DataFrame:
    '''

    :param data: The original non-private dataframe
    :param partitioner: The private partition (Tree with splits)
    :param eps: The privacy budget
    :param delta: The probability of privacy violation
    :param seed: A random seed
    :return:
    '''
    def get_children(n, d, tree_splits):
        return [S for S in tree_splits if S['pnode'] == n and S['pdir'] == d]

    partitioner.get_partition()
    tree_splits = partitioner.tree_splits
    rand = np.random.RandomState(seed)

    X_feat, y_feat, y_classes = partitioner.X_feat, partitioner.y_feat, partitioner.y_classes
    n_ycl = len(y_classes)
    eps_node = eps / max_nodes

    VTYPE = GRB.CONTINUOUS
    model = Model('optimize')

    """ Variables """
    vars = {}
    for split in tree_splits:
        vars[(split['node'], split['dir'])] = model.addVars(list(range(n_ycl)), lb=0, vtype=VTYPE)

    """ Constraints """
    for split in tree_splits:
        n, d = split['node'], split['dir']
        node     = vars[(n, d)]
        ch_nodes = [vars[(s['node'], s['dir'])] for s in get_children(n, d, tree_splits)]
        if len(ch_nodes) > 0:
            for i in range(n_ycl):
                model.addConstr(node[i] == quicksum(chnode[i] for chnode in ch_nodes))

    """ Objective """
    obj = None
    for split in tree_splits:
        var = vars[(split['node'], split['dir'])]
        noisy_count = split['count'] + rand.laplace(0, 1 / eps_node, n_ycl)
        for i in range(n_ycl):
            obj += (var[i] - noisy_count[i]) * (var[i] - noisy_count[i])

    """Solve"""
    model.setObjective(obj)
    model.setParam('OutputFlag', False)
    model.optimize()

    for split in tree_splits:
        var = vars[(split['node'], split['dir'])]
        split['noisy_count'] = {}
        for i in range(n_ycl):
            split['noisy_count'][i] = np.round(var[i].x).astype(int)

    # todo: Include partition here

    def get_partition():
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

        pr_lev, _stack_prop, _store_path, paths = 0, [], [], {'path': [], 'counts': []}
        for i, prop in enumerate(tree_splits):
            cr_lev, f, (l, u) = prop['lev'], prop['feature'], prop['bounds']

            # Is leaf
            if len(get_children(prop['node'], prop['dir'], tree_splits)) == 0:
                paths['path'].append(_store_path.copy())
                paths['counts'].append(prop['noisy_count'])

            if cr_lev == pr_lev + 1:  # descend
                pass
            elif cr_lev == pr_lev:  # sibling
                # paths['path'].append(_store_path.copy())
                # paths['counts'].append(prop['noisy_count'])
                ##############
                pop(_stack_prop)
                pop(_store_path)
            elif cr_lev < pr_lev:  # backtrack / backjump
                # paths['path'].append(_store_path.copy())
                # paths['counts'].append(prop['noisy_count'])
                dist = pr_lev - cr_lev + 1
                pop(_stack_prop, dist)
                pop(_store_path, dist)


            push_property(_stack_prop, prop, data)
            _store_path.append((f, l, u))
            # memorize current level info
            pr_lev = cr_lev

        ## Create partition:
        partition = {'partition': [], 'counts': []}
        for i, elem in enumerate(paths['counts']):
            path, counts = paths['path'][i], list(paths['counts'][i].values())
            partition['partition'].append([(x, partitioner.get_bounds(x, path)) for x in X_feat])
            partition['counts'].append(counts)

        return partition

    partition = get_partition()
    return gen_data_from_partition(data, partition, X_feat, y_feat, y_classes, rand, eps=None)

def to_counts(vec):
    vec[vec<0] = 0
    return vec.astype(int)

def gen_data_from_partition(data, partition, X_feat, y_feat, y_classes, rand=None, eps=None):
    d_gen = pd.DataFrame()
    n = len(partition['partition'])
    for i in range(n):
        pslice, y_counts = partition['partition'][i], partition['counts'][i]

        if eps is not None:
            y_counts += np.round(rand.laplace(0, 1/eps, size=len(y_counts))).astype(int)
            y_counts = to_counts(y_counts)

        _d_gen = {v: [] for v in X_feat + [y_feat]}
        rows_to_generate = np.sum(y_counts)

        distr = rand.uniform
        for xfeat, (l, u) in pslice:
            xtype = type(data[xfeat].values[0])
            # L = self.rand.beta(0.5, 0.5, size=rows_to_generate) * (u - l) + l
            # mu, sigma = (u-l)/2, np.sqrt((u-l)/2)
            # L = self.rand.normal(mu, sigma, size=rows_to_generate)
            # L[L > u] = u
            # L[L < l] = l
            # _d_gen[key] += list(L)
            _d_gen[xfeat] += list(distr(low=l, high=u, size=rows_to_generate).astype(xtype))

        ytype = type(data[y_feat].values[0])
        for i, num_rows in enumerate(y_counts):
            l, u = y_classes[i]
            if num_rows > 0:
                _d_gen[y_feat] += list(distr(low=l, high=u, size=num_rows).astype(ytype))

        d_gen = d_gen.append(pd.DataFrame(_d_gen), sort=False)

    return d_gen
