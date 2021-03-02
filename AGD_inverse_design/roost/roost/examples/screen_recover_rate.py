import argparse
import sys

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AGD')

    parser.add_argument('--Oracle_candidates_filepath',
                        default='/home/glard/AML/roost/roost/examples/roost_simple_candidate_exp1_M0_0224.csv',
                        help='Oracle Model Screened candidates')
    parser.add_argument('--Exp2_BS_candidates_filepath',
                        default='/home/glard/AML/roost/roost/examples/roost_simple_candidate_exp2_bs_M10001.csv',
                        help='Oracle Model Screened candidates')
    parser.add_argument('--Exp2_AL_candidates_filepath',
                        default='/home/glard/AML/roost/roost/examples/roost_simple_candidate_exp2_bo_M10016.csv',
                        help='Oracle Model Screened candidates')
    parser.add_argument('--Exp3_BS_candidates_filepath',
                        default='/home/glard/AML/roost/roost/examples/roost_simple_candidate_exp3_base.csv',
                        help='Oracle Model Screened candidates')
    parser.add_argument('--Exp3_AL_candidates_filepath',
                        default='/home/glard/AML/roost/roost/examples/roost_simple_candidate_exp3_bo.csv',
                        help='Oracle Model Screened candidates')

    args = parser.parse_args(sys.argv[1:])
    m1 = pd.read_csv(args.Oracle_candidates_filepath)
    m2 = pd.read_csv(args.Exp2_BS_candidates_filepath)
    m21 = pd.read_csv(args.Exp2_AL_candidates_filepath)
    m3 = pd.read_csv(args.Exp3_BS_candidates_filepath)
    m31 = pd.read_csv(args.Exp3_AL_candidates_filepath)

    set_m1_candidates = set(m1['composition'].unique())
    set_m2_candidates = set(m2['composition'].unique())
    set_m21_candidates = set(m21['composition'].unique())

    set_m3_candidates = set(m3['composition'].unique())
    set_m31_candidates = set(m31['composition'].unique())
    # -----------------------------------------------------------------------------------
    common_1_2 = set_m1_candidates & set_m2_candidates
    common_1_21 = set_m1_candidates & set_m21_candidates
    print(len(common_1_2))
    print(len(common_1_21))

    print(f"Screening Accuracy for exp2 bs is {len(common_1_2)/len(set_m1_candidates)}")

    print(f"Screening Accuracy for exp2 al is {len(common_1_21)/len(set_m1_candidates)}")
    # --------------------------------------------------------------------------------------
    common_1_3 = set_m1_candidates & set_m3_candidates
    common_1_31 = set_m1_candidates & set_m31_candidates
    print(len(common_1_3))
    print(len(common_1_31))

    print(f"Screening Accuracy for exp3 bs is {len(common_1_3)/len(set_m1_candidates)}")

    print(f"Screening Accuracy for exp3 al is {len(common_1_31)/len(set_m1_candidates)}")
