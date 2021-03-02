from pymatgen.io.cif import CifParser
from pymatgen.alchemy.transmuters import CifTransmuter
from pymatgen.transformations.standard_transformations import SubstitutionTransformation, RemoveSpeciesTransformation
from pymatgen.io.cif import CifWriter
from pymatgen import Composition

import pandas as pd
import subprocess
import os
print(os.getcwd())

def find_diff(a, b):
    # input a, b are tuples
    l = []
    for ai,bi in zip(a,b):
        dic  = {}
        if ai != bi:
            dic[ai] = bi
            l.append(dic)
    return l

df_candidate = pd.read_csv('./inverse_design_candidates_mpid.csv')
#df_candidate = df_candidate.sample(n=300, replace=False, random_state=42)
df_us = df_candidate[['id_original', 'composition', 'mpid']]
df_magpie = pd.read_csv('./bandgap-magpie.csv')

for index, row in df_us.iterrows():
    # output CIF file name:
    id_original = row['id_original']
    composition = row['composition']
    mpid = row['mpid']
    name = str(id_original) + '_' + str(composition) + '_' + str(mpid) + '.cif'
    # find pretty_formula in original dataset
    rowno = df_magpie[df_magpie['material_id'] == mpid].index[0]
    old_formula = df_magpie.iloc[rowno]['pretty_formula']

    # read cif file into memory
    # original_name = './CIF_exp3_bo/' + str(mpid) + '.cif'
    # parser = CifParser(original_name)
    print('original composition: ' + str(old_formula))
    print('target composition: ' + str(composition))
    if composition == old_formula:
        # no replacement, find a match in material project
        try:
            #subprocess.call(["cp", './sorted-symmetrized-cifs/*' + str(mpid) + ".cif", "./rf_semi_target/"+name])
            with open("./inverse_design_rediscover.txt",'a+') as f:
                f.write(str(composition) + '\n')
        except:
            continue

    else:
        ori_d = {k: v for k, v in sorted(Composition(old_formula).as_dict().items(), key=lambda item: item[1])}
        tar_d = {k: v for k, v in sorted(Composition(composition).as_dict().items(), key=lambda item: item[1])}

        ori_t = tuple(ori_d)
        tar_t = tuple(tar_d)
        # print('original composition: ' + str(ori_t))
        # print('target composition: ' + str(tar_t))
        replace_syntax = find_diff(ori_t, tar_t)
        trans = []
        for syn in replace_syntax:
            trans.append(SubstitutionTransformation(syn))
        # os.system("cd ./CIF")
        # print(os.getcwd())
        try:

            transmuter = CifTransmuter.from_filenames(['./renamed_cif/' + str(mpid) + ".cif"], trans)
            structures = transmuter.transformed_structures
            #print(structures[0].final_structure)
            w = CifWriter(structures[0].final_structure)
            w.write_file("./inverse_design_target/"+name)
        except:
            continue
print('success!!')

# parser = CifParser('./file_name.cif')
#
# structure = parser.get_structures()[0]
#
# trans = []
# trans.append(SubstitutionTransformation({"F":"Cl"}))
# #trans.append(RemoveSpeciesTransformation(["F"]))
# transmuter = CifTransmuter.from_filenames(["file_name.cif"], trans)
# structures = transmuter.transformed_structures
#
# print(structures[0].final_structure)
#
# w = CifWriter(structures[0].final_structure)
# w.write_file('mystructure.cif')
# print('success')


# import pandas as pd
# import requests
#
# df_candidate = pd.read_csv('./candidate_mpid.csv')
# i = 0
# df_us = df_candidate[['id_original', 'composition', 'mpid']]
# for id_original, composition, mpid in df_us.itertuples(index = False):
#     # download cif
#     name = str(id_original) + '_' + str(composition)
