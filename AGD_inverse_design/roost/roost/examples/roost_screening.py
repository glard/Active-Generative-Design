import sys
sys.path.insert(0, '/home/rxin/AML/bo_bandgap/bandgap_prediction')
import os
path = os.getcwd()
print(path)
print(os.getcwd())
import pandas as pd

import pickle
import numpy as np

import csv
import sys
sys.path.append('/home/glard/AML/roost/roost/examples')
from pymatgen import Composition

from model_inference import inference

if __name__ == '__main__':
    #header_list = ["composition"]
    """
    2. band gap value prediction using roost model
    """
    df_avial = pd.read_csv('./clean_formula_df.csv')
    # print(df_avial["composition"].size)
    # df_avial["id"]= list(range(1,df_avial["composition"].size+1))
    # df_avial["Eg"]= list(range(1,df_avial["composition"].size+1))
    #
    # cols = ["id", "composition", "Eg"]
    # df_avial = df_avial[cols]
    # df_avial.to_csv("clean_formula_df.csv", names=cols)
    bandgap_formula_list = []

    out = inference('./clean_formula_df.csv')
    #bandgap_str = formula+','+str(bandgap[0])
    out = list(out)

    #print(out)
    df_avial["Eg"]=out
    df_avial.to_csv("roost_screening_result_exp1_208_m9.csv", index=False, header=True)

    df_bandgap = pd.read_csv('./roost_screening_result_exp2_127_m211.csv')
    """
    3. discard item whose band gap value < 6
    """
    df_filtered = df_bandgap[df_bandgap["Eg"] >= 6]


    num_atom_l = []
    num_elements_l = []
    for c in df_filtered["composition"]:
        c = Composition(c)
        num_elements = len(c.as_dict())
        num_atom = c.num_atoms
        num_atom_l.append(num_atom)
        num_elements_l.append(num_elements)
    df_filtered["num_atom"] = num_atom_l
    df_filtered["# of elements"] = num_elements_l

    df_filtered = df_filtered[df_filtered["num_atom"] < 8]
    df_filtered = df_filtered[df_filtered["# of elements"] < 4]

    df_filtered.to_csv("roost_simple_candidate_exp1_M9.csv", index=False, header=True)
    print('success!')
