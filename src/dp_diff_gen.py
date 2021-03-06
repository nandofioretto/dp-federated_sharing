# Algorithm description
# DiffGen: differentially private anonymization based on generalization Mohammed et al. [26] proposed DiffGen to
# publish histograms for classification under differential privacy. It consists of 2 steps, partition and perturbation.
# Given a dataset D and taxonomy trees for each predictor attribute, the partition step starts by generalizing all
# attributes’ values into the topmost nodes in their taxonomy trees.
# It then iteratively selects one attribute’s taxonomy tree node at a time for specialization by using the exponential
# mechanism. The quality of each candidate specialization is based on the heuristics used by the decision tree
# constructions, such as information gain and majority class. The partition step terminates after a given number of
# specializations.
# The perturbation step injects Laplace noise into each cell of the partition and outputs all the cells with their
# noisy counts as the noisy synopsis of the data.
# Privacy budget needs to be evenly distributed to all the specialization steps.


# Inputs: dataset, taxonomy (generated by tree), privacy budget, maximum number of splits
import time

import numpy as np
import pandas as pd


def exponential_mechanism(eps, DeltaF, F, Fargs, choices, rand=None):
    pr_vec, xf_vec = [], []
    cuts, children, D, y_feat, n_yclasses, sslice, _ = Fargs
    for xname in choices:
        _Fargs = [cuts, children, D, y_feat, n_yclasses, sslice, xname]
        pr_vec.append(exponential_function(eps, DeltaF, F, _Fargs, choices))
        xf_vec.append(xname)
    if rand is None:
        return np.random.choice(xf_vec, p=pr_vec)
    else:
        return rand.choice(xf_vec, p=pr_vec)

def exponential_function(eps, DeltaF, F, Fargs, choices):
    """
    :param eps: The privacy budget
    :param DeltaF: The sensitiv of the scoring function
    :param F: The scoring function
    :param Fargs: The arguments of the scoring function
    :param choices: The set of possible choices
    :return:
    """
    from scipy.misc import logsumexp
    _Fargs, choice = Fargs[:-1], Fargs[-1]
    vals = {vi: (eps/(2*DeltaF)) * F(_Fargs+[vi]) for vi in choices}
    return np.exp(logsumexp(vals[choice]) - logsumexp(list(vals.values())))
    # Pr = np.exp((eps/(2*DeltaF)) * F(Fargs)) / np.sum([np.exp((eps/(2*DeltaF)) * F(_Fargs+[vi])) for vi in choices])
    # return Pr

def max_score(args):
    """
        args[0] = partition
        args[1] = D
        args[2] = y_feat
        args[3] = n_yclasses=2
        args[4] = sslice=[]
        args[5] = xname
    :return:
    """
    cuts, children, D, y_feat, n_yclasses, sslice, xname = args
    smax = 0
    x_ch_cuts = [cuts[ch] for ch in children[xname]]
    for (cl, cu) in x_ch_cuts:
        _s = sslice + [(D[xname] >= cl) & (D[xname] <= cu)]
        smax += max(count_cl_items(D, _s, y_feat, n_yclasses))
    return smax

def count_cl_items(D, pstack, clname, cl_size):
    return np.bincount(D[np.asarray(pstack).all(axis=0)][clname].values, minlength=cl_size)

def get_partition(partitioner, D):
    """
    Return some like this:
    SepalLength [(4.3, 7.9)]
    SepalWidth [(2.0, 4.4)]
    PetalLength [(1.0, 2.45), (2.45, 4.85), (4.85, 4.95), (4.95, 6.9)]
    PetalWidth [(0.1, 1.65), (1.65, 1.75), (1.75, 2.5)]
    """
    partition = partitioner.get_partition()
    # for p in partition:
    #     print(p)
    x_feat, y_feat, y_classes = partitioner.X_feat, partitioner.y_feat, partitioner.y_classes
    parts = {x: [(D[x].min(), D[x].max())] for x in x_feat}

    for P in partition:
        for a, (l, u) in P:
            if (l, u) in parts[a]: continue
            # find interval which contains (l, u)
            for i, (_l, _u) in enumerate(parts[a]):
                if l >= _l and u <= _u:
                    del parts[a][i]
                    if l > _l: parts[a].append((_l, l))
                    parts[a].append((l, u))
                    if u < _u: parts[a].append((u, _u))
                    break
            parts[a] = sorted(parts[a], key=lambda x: (x[0], x[1]))
    return parts

def get_cuts_and_children(partition):
    # Cuts contains the {id_feature: (lb, ub) \forall possible features and specializations}
    cuts = {}
    # chidren: contains the ch relation for all possible features and specializations
    children = {None: [v for v in partition]}
    map_names = {v: v for v in partition}
    for k, v in partition.items():
        if len(v) == 1: # no children
            children[k] = []
            cuts[k] = v[0]
        else:
            children[k] = [k+str(i) for i,bounds in enumerate(v)]
            cuts[k] = (v[0][0], v[-1][-1])
            for i, ch in enumerate(children[k]):
                children[ch] = []
                cuts[ch] = (v[i][0], v[i][-1])
                map_names[ch] = map_names[k]

    return cuts, children, map_names

def diff_gen2(data, _partition, x_feat, y_feat, y_classes, eps, delta=None, h=15, seed=1234):

    D = data
    parts = {x: [(D[x].min(), D[x].max())] for x in x_feat}

    for P in _partition:
        for a, (l, u) in P:
            if (l, u) in parts[a]: continue
            # find interval which contains (l, u)
            for i, (_l, _u) in enumerate(parts[a]):
                if l >= _l and u <= _u:
                    del parts[a][i]
                    if l > _l: parts[a].append((_l, l))
                    parts[a].append((l, u))
                    if u < _u: parts[a].append((u, _u))
                    break
            parts[a] = sorted(parts[a], key=lambda x: (x[0], x[1]))
    partition = parts


    rand = np.random.RandomState(seed)
    n_yclasses = len(y_classes)
    cuts, children, map_names = get_cuts_and_children(partition)

    def has_children(name):
        return len(children[name]) > 1

    lev0_choices = [pname for pname in partition if has_children(pname)]
    nFeat = len(lev0_choices)
    maxh = np.prod([len(p) for p in partition.values()])
    h = min(h,maxh)
    _eps = eps / (2*nFeat + 2*h)
    F, DeltaF = max_score, 1

    ret_partition_set = {'partition': [], 'counts': []}
    _curr_partition = {feat: cuts[feat] for feat in partition}

    map_exp_values = {l: {} for l in range(h)}

    def recur(level, level_choices, slice_stack=[], choice=None):
        if level == h or not has_children(choice): # OR len(level_choices) == 0
            ret_partition_set['partition'].append([(f, b) for f, b in _curr_partition.items()])
            ret_partition_set['counts'].append(count_cl_items(D, slice_stack, y_feat, n_yclasses))
            # print('leaf!', ret_partition_set['partition'][-1], ret_partition_set['counts'][-1])
            return

        if level == 0: # Root
            # The choices at root level are those which children
            Fargs = [cuts, children, D, y_feat, n_yclasses, slice_stack, None]
            choice = exponential_mechanism(_eps, DeltaF, F, Fargs, level_choices, rand)
            l, u = cuts[choice]
            slice_stack.append((D[map_names[choice]] >= l) & (D[map_names[choice]] <= u))
            nextlev_choices = [c for c in level_choices if c != choice]
            recur(level+1, nextlev_choices, slice_stack, choice)
            slice_stack.pop()
        else:
            Fargs = [cuts, children, D, y_feat, n_yclasses, slice_stack, None]

            ## Precompute all values in:  children[choice] U level_choices
            values = list(set(children[choice] + level_choices))
            map_exp_values[level] = precompute_exp_values(values, _eps, F, Fargs, DeltaF)
            ##############################

            for ch in children[choice]:
                ch_choice = fast_exponential_mechanism(map_exp_values[level], level_choices + [ch], rand)
                #ch_choice = exponential_mechanism(_eps, DeltaF, F, Fargs, level_choices + [ch], rand)
                l, u = cuts[ch]
                slice_stack.append((D[map_names[ch]] >= l) & (D[map_names[ch]] <= u))
                counts = count_cl_items(D, slice_stack, y_feat, n_yclasses)
                _curr_partition[map_names[ch]] = cuts[ch]

                if np.sum(counts) > 2:
                    nextlev_choices = [c for c in level_choices if c != ch_choice]
                    recur(level + 1, nextlev_choices, slice_stack,  ch_choice)
                slice_stack.pop()

            ## flush away map_exp_values for this model
            map_exp_values[level] = {}

        return ret_partition_set

    ret_partition_set = recur(0, lev0_choices)

    from dp_data_generator import gen_data_from_partition
    return gen_data_from_partition(D, ret_partition_set, x_feat, y_feat, y_classes, rand, eps/2)


def diff_gen(data, partitioner, eps, delta=None, h=15, seed=1234):
    D = data
    rand = np.random.RandomState(seed)

    partition = get_partition(partitioner, D)
    x_feat, y_feat, y_classes = partitioner.X_feat, partitioner.y_feat, partitioner.y_classes

    rand = np.random.RandomState(seed)
    n_yclasses = len(y_classes)
    cuts, children, map_names = get_cuts_and_children(partition)

    def has_children(name):
        return len(children[name]) > 1

    lev0_choices = [pname for pname in partition if has_children(pname)]
    nFeat = len(lev0_choices)
    maxh = np.prod([len(p) for p in partition.values()])
    h = min(h,maxh)
    _eps = eps / (2*nFeat + 2*h)
    F, DeltaF = max_score, 1

    ret_partition_set = {'partition': [], 'counts': []}
    _curr_partition = {feat: cuts[feat] for feat in partition}

    map_exp_values = {l: {} for l in range(h)}

    def recur(level, level_choices, slice_stack=[], choice=None):
        if level == h or not has_children(choice): # OR len(level_choices) == 0
            ret_partition_set['partition'].append([(f, b) for f, b in _curr_partition.items()])
            ret_partition_set['counts'].append(count_cl_items(D, slice_stack, y_feat, n_yclasses))
            # print('leaf!', ret_partition_set['partition'][-1], ret_partition_set['counts'][-1])
            return

        if level == 0: # Root
            # The choices at root level are those which children
            Fargs = [cuts, children, D, y_feat, n_yclasses, slice_stack, None]
            choice = exponential_mechanism(_eps, DeltaF, F, Fargs, level_choices, rand)
            l, u = cuts[choice]
            slice_stack.append((D[map_names[choice]] >= l) & (D[map_names[choice]] <= u))
            nextlev_choices = [c for c in level_choices if c != choice]
            recur(level+1, nextlev_choices, slice_stack, choice)
            slice_stack.pop()
        else:
            Fargs = [cuts, children, D, y_feat, n_yclasses, slice_stack, None]

            ## Precompute all values in:  children[choice] U level_choices
            values = list(set(children[choice] + level_choices))
            map_exp_values[level] = precompute_exp_values(values, _eps, F, Fargs, DeltaF)
            ##############################

            for ch in children[choice]:
                ch_choice = fast_exponential_mechanism(map_exp_values[level], level_choices + [ch], rand)
                #ch_choice = exponential_mechanism(_eps, DeltaF, F, Fargs, level_choices + [ch], rand)
                l, u = cuts[ch]
                slice_stack.append((D[map_names[ch]] >= l) & (D[map_names[ch]] <= u))
                counts = count_cl_items(D, slice_stack, y_feat, n_yclasses)
                _curr_partition[map_names[ch]] = cuts[ch]

                if np.sum(counts) > 2:
                    nextlev_choices = [c for c in level_choices if c != ch_choice]
                    recur(level + 1, nextlev_choices, slice_stack,  ch_choice)
                slice_stack.pop()

            ## flush away map_exp_values for this model
            map_exp_values[level] = {}

        return ret_partition_set

    ret_partition_set = recur(0, lev0_choices)

    from dp_data_generator import gen_data_from_partition
    return gen_data_from_partition(D, ret_partition_set, x_feat, y_feat, y_classes, rand, eps/2)



def precompute_exp_values(values, eps, F, Fargs, DeltaF):
    res = {}
    for v in values:
        Fargs[-1] = v
        #res[v] = np.exp((eps / (2 * DeltaF)) * F(Fargs))
        res[v] = (eps / (2 * DeltaF)) * F(Fargs)
    return res

def fast_exponential_mechanism(pr_values, choices, rand=None):
    from scipy.misc import logsumexp

    pr_vec, xf_vec = [], []
    #Z = np.sum(pr_values[y] for y in choices)
    Z = logsumexp(np.asarray([pr_values[y] for y in choices]))
    for xname in choices:
        #pr_vec.append(pr_values[xname] / Z)
        pr_vec.append(np.exp(logsumexp(pr_values[xname]) - Z))
        xf_vec.append(xname)
    if rand is None:
        return np.random.choice(xf_vec, p=pr_vec)
    else:
        return rand.choice(xf_vec, p=pr_vec)


if __name__ == '__main__':
    import pandas as pd
    from sklearn import preprocessing
    from partitioner_tree import TreePartitioner
    from data_loader import load_dataset
    data_path = '/Users/fferdinando3/Repos/differential_privacy/dp-distr-ml/datasets/'
    data = pd.read_csv(data_path + 'fishiris.csv', na_values='?', skipinitialspace=True)
    lb_make = preprocessing.LabelEncoder()
    obj_df = data.select_dtypes(include=['object']).copy()
    for feat in list(obj_df.columns):
        data.loc[:, feat] = lb_make.fit_transform(obj_df[feat])

    x = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    y = 'Name'
    p = 'SepalLength'
    params_fit = {'epsilon': 1,
                  'y-classes': 3,
                  'criterion': 'gini',
                  'max_depth': 15,
                  'min_samples_leaf':1,
                  'min_samples_split':10}

    # data, x, y, p = load_dataset("census", "age")
    tree1 = TreePartitioner(data, x, y, params_fit)
    partition = tree1.get_partition()
    private_data = diff_gen(data=data, partitioner=tree1, h=6, eps=1.0/2)
    print(private_data)
