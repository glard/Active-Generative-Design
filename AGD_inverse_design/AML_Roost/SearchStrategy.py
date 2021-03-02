"""
This file is used to select next sample point more efficiently
"""

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from Utils import TrendPlot
from util import pred_matrix2formula
import csv

class SearchStrategy:
    def __init__(self, boundary, budget, target_func, mode, satisfactory_value, initial_para, initial_target, kappa, outpath):
        self.pbounds = boundary
        self.target_func = target_func
        self.mode = mode
        self.initial_para = initial_para
        self.initial_target = initial_target
        self.kappa = kappa
        self.outpath = outpath
        if self.mode == 2:
            self.budget = 100000000000
            self.satisfactory_value = satisfactory_value
        else:
            self.budget = budget

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 4))])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        model = tf.keras.models.load_model('models/best_autoencoder.h5')
        inputs = model.get_layer('decoder').input
        outputs = model.get_layer('decoder').get_layer('output').output
        self.decoder = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def bo_opt_circle(self):
        # Bounded region of parameter space need to be predefined
        """
        Mode 1 : Given a budget, what is the maximum value bo can achieve
        Mode 2 : How many probes it will need to get a satisfactory value
        :return: none return value
        """

        optimizer = BayesianOptimization(
            f=self.target_func,
            pbounds=self.pbounds,
            verbose=2,
            random_state=42,
        )
        # criteria to choose next sample
        utility = UtilityFunction(kind="ucb", kappa=self.kappa, xi=0.0)
        # add known training set to bo, help it to build a Gaussian Process
        for ip, it in zip(self.initial_para, self.initial_target):
            optimizer.register(
                params=ip,
                target=it,
            )
        # probe next point based on utility function, evaluate next target point, register result
        # list to store num of current max value
        _4plot_max = []

        df_probed = pd.DataFrame([])

        for _ in range(self.budget):
            # successful decode time
            time = 0
            # temp probe candidates vector:
            probe_candidates = []
            # generate 200 probing candidates based on current GP
            latent_vec4register = []
            for i in range(2000):

                next_point_to_probe = optimizer.suggest(utility, train=False)
                print('next point to probe: ', str(next_point_to_probe))
                sorted_probe_list = []
                cols_latent = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
                               'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
                               'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37',
                               'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49',
                               'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61',
                               'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73',
                               'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85',
                               'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97',
                               'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108',
                               'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119',
                               'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127']

                for key in cols_latent:
                    sorted_probe_list.append(next_point_to_probe[key])
                print(len(sorted_probe_list))

                latent_array = np.array([sorted_probe_list])
                print(latent_array.shape)
                matrix = self.decoder.predict(latent_array)[0, :, :, 0]
                matrix = np.rint(matrix)

                try:
                    formula = pred_matrix2formula(matrix)
                    print('Formula recovered: ', formula)
                except:
                    print('incorrect formula detected!')
                    formula = 'Incorrect converted formula'

                if formula != 'Incorrect converted formula':
                    time = time + 1
                    latent_vec4register.append(sorted_probe_list)
                    probe_candidates.append(formula)
                else:
                    optimizer.register(
                        params=sorted_probe_list,
                        target=0.0,
                    )
                    optimizer._gp.fit(optimizer._space.params, optimizer._space.target)

                #print("Next point to probe is:", next_point_to_probe)
                if time >= 20:
                    break
            df_suggestion = pd.DataFrame(probe_candidates, columns= ["composition"])
            df_avial = df_suggestion.copy()
            df_avial["id"] = list(range(1, df_avial["composition"].size + 1))
            df_avial["Eg"]= list(range(1,df_avial["composition"].size+1))
            cols = ["id", "composition", "Eg"]
            df_avial = df_avial[cols]
            try:
                target_candidates = self.target_func(df_avial)
            except:
                continue
            target_candidates = list(target_candidates)

            df_suggestion["Eg"] = target_candidates
            df_probed = df_probed.append(df_suggestion, ignore_index=True)

            for prob, candidate in zip(latent_vec4register, target_candidates):
                optimizer.register(
                    params=prob,
                    target=candidate,
                )

            #update_gp = optimizer.suggest(utility, train=True)
            optimizer._gp.fit(optimizer._space.params, optimizer._space.target)
            record = []
            for form, tar in zip(probe_candidates, target_candidates):
                rec = form + ',' + str(tar)
                record.append(rec)
            with open(self.outpath +'.txt', 'a+') as f:
                f.write('\n')
                f.write('\n'.join(record))

            _4plot_max.append(optimizer.max['target'])
        df_probed.to_csv(self.outpath + '.csv')

        if self.mode == 1:
            TrendPlot.TrendPlot.bst_vs_no(_4plot_max)


        print('\x1b[6;30;42m' + 'the maximum value found is ' + str(optimizer.max) + '\x1b[0m')
        print('\x1b[6;30;42m' + 'the number of probe times is ' + str(self.budget * 20) + '\x1b[0m')

