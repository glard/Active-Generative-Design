import random
random.seed(123)
import numpy as np
np.random.seed(123)
import json
import warnings
import pandas as pd
from pymatgen.core.composition import Composition

def build_entry():
    with open('data/periodic_table.json') as f:
        d = json.load(f)

    f2no = {}#all elements in periodic table
    for e in d.keys():
        f2no[e] = d[e]["Atomic no"]

    df = pd.read_csv('data/mpid_formula_sp.csv')
    formulas = df.pretty_formula.values

    ls = []
    for c in formulas:
        try:
            obj = Composition(c)
            ls += [e.symbol for e in obj.elements]
        except Exception as e:
            pass
       
    elements = list(set(ls)&set(f2no.keys()))

    v2 = {}#all unique elements in Material Project
    for e in elements:
        v2[e] = f2no[e]
    v2 = sorted(v2.items(), key=lambda item: item[1])

    v3 = {}#index re-encoding
    for i, e in enumerate(v2):
        v3[e[0]] = i
    return v3


# revised for some formula has a single element surpass 8
def formula2onehot_matrix(formula,l=8):
    onehot = build_entry()
    obj = Composition(formula)
    d = obj.as_dict()
    if max(d.values()) <= l:
        matrix = np.zeros((len(onehot), l))
        for symbol in d.keys():
            matrix[onehot[symbol],int(d[symbol])-1] = 1
        matrix = np.expand_dims(matrix, -1)
        matrix = np.expand_dims(matrix, 0)
        return matrix
    else:
        warnings.warn('The number of single element in composition '
                              'can not surpass {}.' .format(l))
        return None


def pred_matrix2formula(onehot_matrix):
    symbol2id = build_entry()
    id2symbol = {num:s for s,num in symbol2id.items()}
    temp = ''
    for i, row in enumerate(onehot_matrix):
        if row.sum() != 0.0:
            count = np.where(row==1.0)[0]
            if len(count) == 1:
                temp += id2symbol[i]+str(count[0]+1)
            else:
                return "Incorrect converted formula"
    obj = Composition(temp)
    return obj.reduced_formula


def get_onehot_matrix(l=8):
    onehot = build_entry()
    df = pd.read_csv('data/mpid_formula_sp.csv')
    formulas = df.pretty_formula.values
    formulas = list(set(formulas))
    data = {}
    for c in formulas:
        if isinstance(c,float):
            continue
        obj = Composition(c)
        d = obj.as_dict()
        if max(d.values()) <= l:
            matrix = np.zeros((len(onehot), l))
            for symbol in d.keys():
                matrix[onehot[symbol],int(d[symbol])-1] = 1
            matrix = np.expand_dims(matrix, -1)
            data[c] = matrix

    return data

# modified
def split_data(trn_ratio=0.8,l=8):
    d = get_onehot_matrix(l)
    formulas = list(d.keys())
    random.shuffle(formulas)
    trn_formulas = formulas[:int(len(formulas)*trn_ratio)]
    tst_formulas = formulas[int(len(formulas)*trn_ratio):]

    return trn_formulas, tst_formulas, d

def load_data():
    trn_formulas, tst_formulas, d = split_data()

    X = []
    for f in trn_formulas:
        X.append(d[f])

    val_X = []
    for f in tst_formulas:
        val_X.append(d[f])

    return np.array(X), np.array(val_X)

def load_data_dict():
    trn_formulas, tst_formulas, d = split_data()

    d1 = {}
    for f in trn_formulas:
        d1[f] = d[f]

    d2 = {}
    for f in tst_formulas:
        d2[f] = d[f]
    return d1, d2


    
if __name__ == '__main__':
    pred_matrix2formula(None)


























