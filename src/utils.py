import numpy as np
import pandas as pd
import numpy as np
import sys
import os
import errno

def create_dir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def to_probability(x):
    return x / sum(x)

def scale(x, min, max):
    return (max - min) * (x - np.min(x)) / (np.max(x) - np.min(x)) + min

def normalize(x, min, max):
    if not is_normalized(x, min, max):
        return (max - min) * (x - np.min(x)) / (np.max(x) - np.min(x)) + min
    else:
        return x

def to_multinomial(y, nclasses):
    cat, vals = is_categorical_(y)
    if cat:
       if len(vals) <= nclasses:
           classes = dict({i: (v, v) for i, v in enumerate(vals)})
       else:
           d = np.linspace(y.values.min(), y.values.max(), nclasses + 1).astype(int)
           classes = dict({i: (d[i], d[i + 1]) for i in range(len(d) - 1)})
    else:
        d = np.linspace(y.values.min(), y.values.max(), nclasses + 1)
        classes = dict({i: (d[i], d[i + 1]) for i in range(len(d) - 1)})

    _y = y.copy()
    for k, val in enumerate(y.values):
        for cl, (a, b) in classes.items():
            if a <= val <= b:
                _y.values[k] = cl
                break
    return _y, classes

def is_normalized(x, lb, ub):
    return np.all(x <= ub) & np.all(x >= lb)

def is_categorical_(y):
    a = np.unique(y.values.flatten())
    lb, ub = np.min(a), np.max(a)
    if a.dtype == np.int:
        # check for categorical
        if len(a) == ub - lb + 1:
            return True, a
    return False, (lb, ub)

def get_cat_indexes(_y, y_classes):
    '''Process the output of the >to_multinomial< function to return a dictionairy
        of elements
        {cl: [idx] }
        with cl the ID of the class in y_classes, and [idx] the array of indexes of rows in _y
        whose value equal cl.
    '''
    _y_cl = {}
    for i in y_classes.keys():
        _y_cl[i] = np.where(_y.values == i)[0]
    return _y_cl

def is_categorical(data: pd.DataFrame, cl: str):
    '''
    Check if attribute 'cl' of 'data' is categorical or continuous.

    :param data: A pandas DataFrame
    :param cl: An attribute of data
    :return:
        - 1st element: True / False (for categorical or continuous)
        - 2nd element: The array of values if it is categorical
                        min/max value, if continuous.
    '''
    ''' 
    Returns if cl is cateogorical and the numbe of classes
        If it is continous return the support
    '''

    a = np.unique(data[[cl]].values.flatten())
    lb, ub = np.min(a), np.max(a)
    if a.dtype == np.int:
        # check for categorical
        if len(a) == ub - lb + 1:
            return True, a
    return False, (lb, ub)
