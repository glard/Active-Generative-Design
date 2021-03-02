# this file is to get a picture of band gap dataset

import pandas as pd

df = pd.read_csv('../bandgap_df_whole.csv')

df =df.drop_duplicates(subset=[i for i in df.columns if i not in ['band_gap']],
                                           keep='first')

sorted_df = df.sort_values(by=['band_gap'])
y = sorted_df['band_gap']

med = y.median(axis = 0)
max_y = y.max(axis = 0)
top10 = y.iloc[round(sorted_df.shape[0] * 0.9)]

top3_4 = y.iloc[round(sorted_df.shape[0] * 0.966)]
top3 = y.iloc[round(sorted_df.shape[0] * 0.97)]
top1 = y.iloc[round(sorted_df.shape[0] * 0.99)]
top5 = y.iloc[round(sorted_df.shape[0] * 0.95)]
print('median label (50% band_gap) is ' + str(med))
print('max label (top 1 band_gap) is ' + str(max_y))
print('target label (90% band_gap) is ' + str(top10))
print('top 5% label (top 1 band_gap) is ' + str(top5))

print('top 3.4% label (top 1 band_gap) is ' + str(top3_4))
print('top 3% label (top 1 band_gap) is ' + str(top3))
print('top 1% label (top 1 band_gap) is ' + str(top1))