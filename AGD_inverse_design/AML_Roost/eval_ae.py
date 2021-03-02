from util import formula2onehot_matrix
import os.path
from encoder import get_latent_space
from util import pred_matrix2formula, build_entry
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
import random
import tensorflow as tf

onehot = build_entry()
#df = pd.read_csv('data/mpid_formula_sp.csv')
df = pd.read_csv('./cleaned_formula.csv')
formulas = df.pretty_formula.values
formulas = list(set(formulas))
data = {}
for c in formulas:
    if isinstance(c,float):
        continue
    obj = Composition(c)
    d = obj.as_dict()
    if max(d.values()) <= 8:
        matrix = np.zeros((len(onehot), 8))
        for symbol in d.keys():
            matrix[onehot[symbol],int(d[symbol])-1] = 1
        matrix = np.expand_dims(matrix, -1)
        data[c] = matrix
pretty_formula = random.sample(list(data), 5000)
# >>> keys
# [52, 3, 10, 92, 86, 42, 99, 73, 56, 23]
onehot_from_formula = [data[k] for k in pretty_formula]
model = tf.keras.models.load_model('models/best_autoencoder.h5')
inputs = model.get_layer('decoder').input
outputs = model.get_layer('decoder').get_layer('output').output
decoder = tf.keras.models.Model(inputs=inputs, outputs=outputs)
fail_num = 0
suc_num = 0
for p in pretty_formula:
    # encoding
    onehot_matrix = formula2onehot_matrix(p, l=8)
    if onehot_matrix is not None:
        #print(onehot_matrix.shape)
        lat_vec = get_latent_space(onehot_matrix)
        #np.save('data/Sr2YMn2AlO7.npy', lat_vec)
        #print(lat_vec)

        # decoding
        #print(lat_vec.shape)


        matrix = decoder.predict(lat_vec)[0, :, :, 0]
        matrix = np.rint(matrix)

        try:
            print('------------------------------------------')
            formula = pred_matrix2formula(matrix)
            print('Formula recovered: ', formula)
            if Composition(formula) != Composition(p):
                fail_num += 1
                print('decode failed; original input is {}'+ (str(p)))
                print('currect failing time: ' + (str(fail_num)))
            else:
                suc_num += 1
                print('decode success!')
                print('currect success time: ' + (str(suc_num)))
        except:
            print('incorrect formula detected!')
            formula = 'Incorrect converted formula'
            fail_num += 1


acc = 1-fail_num/5000
print("accuracy is " + (str(acc)))
