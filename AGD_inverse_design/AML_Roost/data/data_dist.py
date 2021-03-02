# this file is used to plot band gap dataset distribution

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../bandgap_df_whole.csv')

bandgap_s = df['band_gap']

bandgap_s.plot.hist(grid=True,bins=200, rwidth=0.8,
                    color='#607c8e')
plt.title('Distribution of band gaps of the training set', fontsize=15)
plt.xlabel('band gap (eV)', fontsize=12)
plt.ylabel('Number of Occurences', fontsize=12)
plt.xlim((0.0, 10.0))
#plt.show()
plt.savefig('Dist_bandgap_dataset.pdf')
print('success')