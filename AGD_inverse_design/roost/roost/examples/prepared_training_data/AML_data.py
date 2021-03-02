import pandas as pd
from pymatgen.core.composition import Composition

bd_aml_whole_train = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/bd_AML_whole_train.csv')

bd_aml_whole_train = bd_aml_whole_train.sample(n=5805, replace=True, random_state=42)
bd_aml_whole_train.to_csv('./bd_random5805.csv', index=False, header=True, columns=["id","composition","Eg"])

bd_0_3k = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/bd_AML_whole_init_300.csv',
                      usecols= ['id', 'composition', 'Eg'])
bd_1k = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/bd_AML_whole_init_1000.csv',
                    usecols= ['id', 'composition', 'Eg'])

bd_AML_0_3k = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/exp3_roost_recommandation_budget1000_initial300_kappa100_121.csv')
bd_AML_0_3k['id'] = list(range(1, bd_AML_0_3k["composition"].size + 1))
bd_AML_0_3k['id'] = bd_AML_0_3k['id'] + 99999

bd_AML_1k = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/exp3_roost_recommandation_budget10000_initial1000_kappa100_127.csv')
bd_AML_1k['id'] = list(range(1, bd_AML_1k["composition"].size + 1))
bd_AML_1k['id'] = bd_AML_1k['id'] + 99999

cols = bd_AML_0_3k.columns.tolist()
cols = cols[-1:] + cols[:-1]
bd_AML_0_3k = bd_AML_0_3k[cols]
if bd_AML_0_3k.columns.tolist() == bd_0_3k.columns.tolist():
    print(f'True')
else:
    print(str(bd_AML_0_3k.columns.tolist()))
    print(str(bd_0_3k.columns.tolist()))
bdBOOST_4_train_300p = pd.concat([bd_0_3k,bd_AML_0_3k])
bdBOOST_4_train_1000p = pd.concat([bd_1k,bd_AML_1k])


avail_formula_list = []
id_list = []
eg_list = []
for id, formula, Eg in zip(bdBOOST_4_train_1000p["id"],  bdBOOST_4_train_1000p["composition"],  bdBOOST_4_train_1000p["Eg"]):
    try:
        p = Composition(formula)
        if (len(p.as_dict()) > 1.0):
            avail_formula_list.append(formula)
            id_list.append(id)
            eg_list.append(Eg)
    except:
        continue
# with open('./bd_AML_30_BOOST_118.csv', 'w') as f:
#     f.write('\n'.join(avail_formula_list))
d = {'id': id_list, 'composition': avail_formula_list, 'Eg': eg_list}
df = pd.DataFrame(data=d)
df = df.drop_duplicates(subset='composition',
                                           keep='first')
df.to_csv('./bdBOOST_train_init1000.csv', index=False, header=True, columns=["id","composition","Eg"])
bd_1k.to_csv('bd_original_train_init1000.csv', index=False, header=True, columns=["id","composition","Eg"])


