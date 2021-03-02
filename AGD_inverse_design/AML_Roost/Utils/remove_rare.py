from util import build_entry
import os
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
import re
path = os.getcwd()
print(path)
os.chdir('../')


onehot = build_entry()
print(onehot)
onehot_l = list(onehot.keys())
print(onehot_l)

filepath = './Utils/bandgap-magpie.csv'
df = pd.read_csv(filepath)
#df = df.sample(frac=0.001, replace=True, random_state=1)
print('The shape of current dataset is ' + str(df.shape))

added_columns_name = []
for i in range(128):
    added_columns_name.append('V'+str(i))
data = []
# create composition column
df_comp = StrToComposition(target_col_id='composition').featurize_dataframe(df, 'pretty_formula')
# create column with maximum atom number
max_atom_num = []
for st in df_comp[['composition']].astype(str).values:
    atom_list = []
    #print(st[0])
    s = st[0]
    for item in s.split():
        num = re.sub(r"\D", "", item)
        atom_list.append(int(num))
    #print(atom_list)
    max_atom_num.append(max(atom_list))

# update dataframe with max_atom_num
df['max_atom_num'] = max_atom_num
# remove rows whose max atom number above 20
df = df[df['max_atom_num'] < 21]

# remove record whose composition has key not in list
print(df_comp['pretty_formula'].str.contains('H|He|Li|Be|B|C|N|O|F|Na|Mg|Al|Si|P|S|Cl|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Ac|Th|Pa|U|Np|Pu'))
#index = [df_comp['pretty_formula'].str.contains('H|He|Li|Be|B|C|N|O|F|Na|Mg|Al|Si|P|S|Cl|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Ac|Th|Pa|U|Np|Pu')]
df1 = df[df_comp['pretty_formula'].str.contains('H|He|Li|Be|B|C|N|O|F|Na|Mg|Al|Si|P|S|Cl|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Ac|Th|Pa|U|Np|Pu')]
# validate removing
print('The shape of unprocessed dataset is ' + str(df.shape))
print('The shape of processed dataset is ' + str(df1.shape))
df1 = df1.drop(['max_atom_num'], axis=1)
df1.to_csv(r'bandgap_df_removed_rare.csv', index=False, header=True)