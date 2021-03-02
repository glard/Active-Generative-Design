"""
This file is used to conduct active learning experiment
"""
import argparse
import sys

import SearchStrategy
import preprocessing
import numpy as np
import model_inference

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AGD')
    parser.add_argument('--budget', default=50,
                        help='Active learning budget')
    parser.add_argument('--filepath', default='./Utils/bandgap-magpie.csv',
                        help='Prepared dataset used for active learning')
    parser.add_argument('--dataset', default='bandgap',
                        help='dataset name')
    parser.add_argument('--kappa', default=100,
                        help='hyperparameter to control exploration and exploitation of bayesian optimization')
    parser.add_argument('--candidate_out_path', default='./Exp3_candidates.txt',
                        help='candidate out path')
    parser.add_argument('--init', default=300,
                        help='Active learning initial samples')

    args = parser.parse_args(sys.argv[1:])

    # AGD --- parameters
    budget = args.budget
    filepath = args.filepath
    dataset = args.dataset
    kappa = args.kappa
    candidate_out = args.candidate_out_path
    init_samples = args.init

    p = preprocessing.preprocessing(filepath, dataset, init_samples)

    # provide initial sample points for bo to reference
    X, y, top10, cols = p.bo_read_data()

    pbounds = p.get_range_dic()

    function = model_inference.inference

    para = np.asarray(X, dtype=np.float32)

    para_dic_in_list = []
    for l in X:
        p_dic = {}
        for key, v in zip(cols, l):
                p_dic[key] = v
        para_dic_in_list.append(p_dic)

    target = np.asarray(y, dtype=np.float32)

    s = SearchStrategy.SearchStrategy(satisfactory_value = top10, target_func = function,
                                           initial_para = para_dic_in_list,
                                           initial_target = target,
                                           mode = 1,
                                           boundary=pbounds, budget = budget, kappa=kappa, outpath = candidate_out)
    s.bo_opt_circle()


