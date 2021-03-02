import pickle
import pandas as pd
from sklearn.model_selection import train_test_split




if __name__ == '__main__':
    header_list = ["Pretty_Formula"]
    df = pd.read_csv('../bandgap_df_whole.csv')
    y = df['band_gap'].tolist()

    cols_latent = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                   'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
                   'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43',
                   'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57',
                   'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71',
                   'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85',
                   'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99',
                   'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111',
                   'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123',
                   'V124', 'V125', 'V126', 'V127']

    cols = [i for i in df.columns if (i not in ['band_gap'] and i not in cols_latent)]
    X = df[cols].values.tolist()
    print('splitting dataset into training and testing; test set is 25%')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # load rf model
    loaded_model = pickle.load(open('initialized_model_RF.sav', 'rb'))
    print(len(x_test))
    print(x_test[0])
    y_predict = loaded_model.predict(x_test[0])

    print('success')