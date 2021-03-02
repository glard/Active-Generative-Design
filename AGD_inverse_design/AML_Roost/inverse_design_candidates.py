import pandas as pd
from pymatgen import Composition
from charge_neutrality_fast import check_neutrality
from electronegativity_check_fast import check_electronegativity

# get active learning data

df1 = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/exp3_roost_recommandation_budget1000_initial300_kappa100_114.csv')

df2 = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/exp3_roost_recommandation_budget1000_initial300_kappa100_119.csv')

df3 = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/exp3_roost_recommandation_budget1000_initial300_kappa100_121.csv')

df4 = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/exp3_roost_recommandation_budget10000_initial1000_kappa100_127.csv')

#df5 = pd.read_csv()

df = pd.concat([df1, df2, df3, df4])
df = df.drop_duplicates(subset = ['composition'], keep='first')
print(f"before Eg value check: {df.shape}")

# get Eg > 5.0 eV and formula check
df = df[df['Eg'] > 5.0]

print(f"After Eg value check: {df.shape}")

# formula physical check
candidate_comp_list = []
candidate_eg_list = []
for comp, eg in zip(df['composition'], df['Eg']):
    if check_electronegativity(comp) & check_neutrality(comp):
        candidate_comp_list.append(comp)
        candidate_eg_list.append(eg)


df_candi = pd.DataFrame()
df_candi['composition'] = candidate_comp_list
df_candi['Eg'] = candidate_eg_list

df_candi.to_csv('./inverse_design_candidates.csv')

