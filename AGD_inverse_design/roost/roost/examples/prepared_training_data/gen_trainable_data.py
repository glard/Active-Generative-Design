import pandas as pd
from pymatgen.core.composition import Composition
from sklearn.model_selection import train_test_split



bd_AML_30_BOOST = pd.read_csv('/home/glard/AML/roost/roost/examples/prepared_training_data/bandgap4new_model.csv')
bd_AML_30_BOOST = bd_AML_30_BOOST.drop_duplicates(subset=['composition'],
                                           keep='first')

bd_AML_30_BOOST['id'] = list(range(1, bd_AML_30_BOOST["composition"].size + 1))
avail_formula_list = []
id_list = []
eg_list = []
for id, formula, Eg in zip(bd_AML_30_BOOST["id"],  bd_AML_30_BOOST["composition"],  bd_AML_30_BOOST["Eg"]):
    try:
        p = Composition(formula)
        if (len(p.as_dict()) > 1.0 and all(x<9 for x in p.as_dict().values())):
            avail_formula_list.append(formula)
            id_list.append(id)
            eg_list.append(Eg)
    except:
        continue
# with open('./bd_AML_30_BOOST_118.csv', 'w') as f:
#     f.write('\n'.join(avail_formula_list))
d = {'id': id_list, 'composition': avail_formula_list, 'Eg': eg_list}
df = pd.DataFrame(data=d)
df.to_csv('./bd_AML_whole_for_autoencoder.csv', index=False, header=True, columns=["id","composition","Eg"])

# train, test = train_test_split(df, test_size=0.2)
# train.to_csv('./bd_AML_whole_train.csv', index=False, header=True, columns=["id","composition","Eg"])
# test.to_csv('./bd_AML_whole_test.csv', index=False, header=True, columns=["id","composition","Eg"])


#bd_AML_30_BOOST = bd_AML_30_BOOST.loc[len((Composition(bd_AML_30_BOOST['composition']).as_dict())) > 1.0]
bd_AML_30_BOOST['id'] = list(range(1, bd_AML_30_BOOST["composition"].size + 1))
#bd_AML_30_BOOST['id'] = bd_AML_30_BOOST['id'] + 99999
#bd_AML_30_BOOST.to_csv('./bd_AML_30_BOOST.csv', index=False, header=True, columns=["id","composition","Eg"])

print(bd_AML_30_BOOST.shape)
print(df.shape)
# print(train.shape)
# print(test.shape)