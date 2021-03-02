"""
This file is used to do data preprocessing
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from util import formula2onehot_matrix
import os.path
from encoder import get_latent_space
from matminer.featurizers.conversions import StrToComposition
import re


class preprocessing:
    def __init__(self, filepath, dataset, init_samples):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath, usecols= ['material_id', 'pretty_formula', 'band_gap'])
        self.dataset = dataset
        self.init_samples = init_samples
        self.init_filename = './ALSearch_init_'+ str(init_samples) + '.csv'
        if dataset is 'bandgap':
            #self.df = pd.read_csv('./bandgap_df_whole.csv')
            if os.path.exists(self.init_filename) is False:
                # small examples for debugging
                self.df = self.df.sample(n=self.init_samples, replace=True, random_state=42)
                added_columns_name = []
                for i in range(128):
                    added_columns_name.append('V' + str(i))
                data = []

                # convert formula to latent vector
                for formula in self.df['pretty_formula']:
                    print(formula)
                    onehot_matrix = formula2onehot_matrix(formula, l=20)
                    lat_vec = get_latent_space(onehot_matrix)
                    lat_list = lat_vec.tolist()
                    data.append(lat_list[0])
                    print(formula + 'has been converted into latent vector~')

                df_added = pd.DataFrame(data, columns=added_columns_name)
                self.df.reset_index(drop=True, inplace=True)
                df_added.reset_index(drop=True, inplace=True)
                self.df = pd.concat([self.df, df_added], axis=1)


                # rename columns to eliminate ' '
                column_rename = ['id', 'composition', 'Eg', 'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
                                 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31',
                                 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43',
                                 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55',
                                 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67',
                                 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79',
                                 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91',
                                 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103',
                                 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114',
                                 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
                                 'V126', 'V127']
                self.df = self.df.set_axis(column_rename, axis=1, inplace=False)
                #self.df = self.df.drop()

                self.df.to_csv(self.init_filename, index=False, header=True)

            else:
                self.df = pd.read_csv(self.init_filename)
        print('The shape of initial dataset is ' + str(self.df.shape))
        self.label = ['Eg']

        # drop duplicate values
        self.df =self.df.drop_duplicates(subset=[i for i in self.df.columns if i not in self.label],
                                           keep='first')
        print('The shape of init dataset after dropping duplicates is ' + str(self.df.shape))

        self.df = self.df.dropna()

        # sort dataframe by y value
        self.sorted_df = self.df.sort_values(by=self.label)

        #self.test_size = 0.2



    def bo_read_bandgap_data(self):
        y = self.sorted_df['Eg']
        #cols = [i for i in self.sorted_df.columns if i not in self.label]
        cols = ['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41','V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59','V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77','V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127']
        med = y.median(axis = 0)
        max_y = y.max(axis = 0)
        top10 = y.iloc[round(self.sorted_df.shape[0] * 0.9)]
        top3_4 = y.iloc[round(self.sorted_df.shape[0] * 0.966)]
        top2 = y.iloc[round(self.sorted_df.shape[0] * 0.98)]
        top1 = y.iloc[int(self.sorted_df.shape[0]) -1]
        print('median label (50% band_gap) is ' + str(med))
        print('max label (top 1 band_gap) is ' + str(max_y))
        print('target label (90% band_gap) is ' + str(top10))
        print('top 3.4% label (top 1 band_gap) is ' + str(top3_4))
        print('top 2% label (top 1 band_gap) is ' + str(top2))
        print('top 1% label (top 1 band_gap) is ' + str(top1))


        # select  ys from bottom 50 percent
        #df_bottom = self.sorted_df.loc[(self.sorted_df['band_gap'] < med)]

        # caution!!!!!!!!!!!!!!!!!!! here is a bug caused by df.sample method!!!!!!!!!!!!!!!!!!!!!!!
        #df_bottom = df_bottom.sample(frac=1.0, replace=False, random_state=1)

        # remove duplicate
        df = self.sorted_df.drop_duplicates(subset=cols,
                                     keep='first')
        df = df[df['band_gap'] <= 3.0 ]

        df = self.sorted_df.sample(n=300, replace=False, random_state=42)
        #df = self.sorted_df
        df.to_csv('t-sne-train.csv')
        y = df['Eg'].tolist()
        X = df[cols].values.tolist()
        print('selecting all samples from bandgap dataset.....')

        return X, y, top10, cols


    def bo_read_data(self):
        return self.bo_read_bandgap_data()


    def get_range_dic(self):
        #df1 = self.df.drop(self.label, axis=1)
        df1 = self.df[['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39','V40','V41','V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52','V53','V54','V55','V56','V57','V58','V59','V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74','V75','V76','V77','V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94','V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127']]
        d = {k: (df1[k].min(axis=0), df1[k].max(axis=0)) for k in
             df1.columns}
        return d



