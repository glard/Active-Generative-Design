from pymatgen.io.cif import CifParser
from pymatgen.alchemy.transmuters import CifTransmuter
from pymatgen.transformations.standard_transformations import SubstitutionTransformation, RemoveSpeciesTransformation
from pymatgen import Composition
from ElMD import ElMD
import pandas as pd

if __name__ == '__main__':
    # calculate difference between screening results with materials project candidates
    # x = ElMD("CaTiO3")
    #
    # t = x.elmd("SrTiO3")
    # # screen condition1--find smallest distance
    # print(t)
    #
    # formula1 = 'SrTiO3'
    # c = Composition(formula1)
    # ag1 = c.as_dict().values()
    #
    # formula2 = 'SrTiO3'
    # c = Composition(formula2)
    # ag2 = c.as_dict().values()
    # print(ag2)
    # # screen condition2--find two dictionaries have same value set
    # print(set(ag1) == set(ag2))
    #
    # file = 'SrTiO3_mp-5229_primitive.cif'
    # parser = CifParser(file)
    # structure = parser.get_structures()[0]
    # print(structure)
    #
    # trans = []
    # trans.append(SubstitutionTransformation({"Fe":"Mn"}))
    # trans.append(RemoveSpeciesTransformation(["Lu"]))
    # transmuter = CifTransmuter.from_filenames(["SrTiO3_mp-5229_primitive.cif"], trans)
    # structures = transmuter.transformed_structures
    #print(structure[0])

    #df_candidate = pd.read_csv('./rf_semiconductor_candidate.csv')
    df_candidate = pd.read_csv('./inverse_design_candidates.csv')
    #df_candidate = df_candidate.sample(n=300, replace=False, random_state=42)
    df_magpie = pd.read_csv('./bandgap-magpie.csv')
    tar_mpid_list = []
    df_t = df_magpie[['material_id','pretty_formula']]
    for candidate in df_candidate['composition']:
        # value set
        can_set = set(Composition(candidate).as_dict().values())
        # composition measurement
        x = ElMD(candidate)
        diff_min = 9999.0
        tar_mpid = 'NOT FOUND'

        for mpid, target in df_t.itertuples(index=False):
            # value set
            tar_set = set(Composition(target).as_dict().values())
            if tar_set == can_set:
                # composition measurement
                diff = x.elmd(target)
                if diff < diff_min:
                    diff_min = diff
                    tar_mpid = mpid
        tar_mpid_list.append(tar_mpid)
    df_candidate['mpid'] = tar_mpid_list
    #df_candidate.columns = ['id','id_original','composition','Eg','num_atom','# of elements','mpid']
    df_candidate.to_csv('inverse_design_candidates_mpid.csv',index=True, header=True)
    print('success!')
