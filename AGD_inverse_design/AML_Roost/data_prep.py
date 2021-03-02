import pandas as pd
from util import formula2onehot_matrix
from encoder import get_latent_space
from matminer.featurizers.conversions import StrToComposition
import re

class prep:
	def __init__(self):
		self.filepath = './Utils/bandgap-magpie.csv'
		self.df = pd.read_csv(self.filepath)
		# drop duplicate values
		print('The shape of whole dataset before dropping duplicates is ' + str(self.df.shape))

		self.df = self.df.drop_duplicates(subset=['pretty_formula'],
										  keep='first')
		print('The shape of whole dataset after dropping duplicates is ' + str(self.df.shape))

		self.df = self.df.sample(frac=0.0001, replace=True, random_state=1)
		added_columns_name = []
		for i in range(128):
			added_columns_name.append('V' + str(i))
		data = []
		# create composition column
		df_comp = StrToComposition(target_col_id='composition').featurize_dataframe(self.df, 'pretty_formula')
		# create column with maximum atom number
		max_atom_num = []
		for st in df_comp[['composition']].astype(str).values:
			atom_list = []
			s = st[0]
			for item in s.split():
				num = re.sub(r"\D", "", item)
				atom_list.append(int(num))
			max_atom_num.append(max(atom_list))

		# update dataframe with max_atom_num
		self.df['max_atom_num'] = max_atom_num
		# remove rows whose max atom number above 8
		self.df = self.df[self.df['max_atom_num'] < 9]
		# convert formula to latent vector
		for formula in self.df['pretty_formula']:
			print(formula)
			onehot_matrix = formula2onehot_matrix(formula, l=8)
			lat_vec = get_latent_space(onehot_matrix)
			lat_list = lat_vec.tolist()
			data.append(lat_list[0])
			print(formula + 'has been converted into latent vector~')

		df_added = pd.DataFrame(data, columns=added_columns_name)
		self.df.reset_index(drop=True, inplace=True)
		df_added.reset_index(drop=True, inplace=True)
		self.df = pd.concat([self.df, df_added], axis=1)

		# perform autoencode to pretty formula
		column_to_remove = ['material_id', 'max_atom_num']

		# generate column names
		self.df = self.df.drop(column_to_remove, axis=1)

		# rename columns to eliminate ' '
		column_rename = ['pretty_formula', 'band_gap', 'MagpieData_minimum_Number', 'MagpieData_maximum_Number',
				 'MagpieData_range_Number', 'MagpieData_mean_Number', 'MagpieData_avg_dev_Number',
				 'MagpieData_mode_Number', 'MagpieData_minimum_MendeleevNumber',
				 'MagpieData_maximum_MendeleevNumber', 'MagpieData_range_MendeleevNumber',
				 'MagpieData_mean_MendeleevNumber', 'MagpieData_avg_dev_MendeleevNumber',
				 'MagpieData_mode_MendeleevNumber', 'MagpieData_minimum_AtomicWeight',
				 'MagpieData_maximum_AtomicWeight', 'MagpieData_range_AtomicWeight',
				 'MagpieData_mean_AtomicWeight', 'MagpieData_avg_dev_AtomicWeight',
				 'MagpieData_mode_AtomicWeight', 'MagpieData_minimum_MeltingT',
				 'MagpieData_maximum_MeltingT', 'MagpieData_range_MeltingT', 'MagpieData_mean_MeltingT',
				 'MagpieData_avg_dev_MeltingT', 'MagpieData_mode_MeltingT', 'MagpieData_minimum_Column',
				 'MagpieData_maximum_Column', 'MagpieData_range_Column', 'MagpieData_mean_Column',
				 'MagpieData_avg_dev_Column', 'MagpieData_mode_Column', 'MagpieData_minimum_Row',
				 'MagpieData_maximum_Row', 'MagpieData_range_Row', 'MagpieData_mean_Row',
				 'MagpieData_avg_dev_Row', 'MagpieData_mode_Row', 'MagpieData_minimum_CovalentRadius',
				 'MagpieData_maximum_CovalentRadius', 'MagpieData_range_CovalentRadius',
				 'MagpieData_mean_CovalentRadius', 'MagpieData_avg_dev_CovalentRadius',
				 'MagpieData_mode_CovalentRadius', 'MagpieData_minimum_Electronegativity',
				 'MagpieData_maximum_Electronegativity', 'MagpieData_range_Electronegativity',
				 'MagpieData_mean_Electronegativity', 'MagpieData_avg_dev_Electronegativity',
				 'MagpieData_mode_Electronegativity', 'MagpieData_minimum_NsValence',
				 'MagpieData_maximum_NsValence', 'MagpieData_range_NsValence',
				 'MagpieData_mean_NsValence', 'MagpieData_avg_dev_NsValence',
				 'MagpieData_mode_NsValence', 'MagpieData_minimum_NpValence',
				 'MagpieData_maximum_NpValence', 'MagpieData_range_NpValence',
				 'MagpieData_mean_NpValence', 'MagpieData_avg_dev_NpValence',
				 'MagpieData_mode_NpValence', 'MagpieData_minimum_NdValence',
				 'MagpieData_maximum_NdValence', 'MagpieData_range_NdValence',
				 'MagpieData_mean_NdValence', 'MagpieData_avg_dev_NdValence',
				 'MagpieData_mode_NdValence', 'MagpieData_minimum_NfValence',
				 'MagpieData_maximum_NfValence', 'MagpieData_range_NfValence',
				 'MagpieData_mean_NfValence', 'MagpieData_avg_dev_NfValence',
				 'MagpieData_mode_NfValence', 'MagpieData_minimum_NValence',
				 'MagpieData_maximum_NValence', 'MagpieData_range_NValence', 'MagpieData_mean_NValence',
				 'MagpieData_avg_dev_NValence', 'MagpieData_mode_NValence',
				 'MagpieData_minimum_NsUnfilled', 'MagpieData_maximum_NsUnfilled',
				 'MagpieData_range_NsUnfilled', 'MagpieData_mean_NsUnfilled',
				 'MagpieData_avg_dev_NsUnfilled', 'MagpieData_mode_NsUnfilled',
				 'MagpieData_minimum_NpUnfilled', 'MagpieData_maximum_NpUnfilled',
				 'MagpieData_range_NpUnfilled', 'MagpieData_mean_NpUnfilled',
				 'MagpieData_avg_dev_NpUnfilled', 'MagpieData_mode_NpUnfilled',
				 'MagpieData_minimum_NdUnfilled', 'MagpieData_maximum_NdUnfilled',
				 'MagpieData_range_NdUnfilled', 'MagpieData_mean_NdUnfilled',
				 'MagpieData_avg_dev_NdUnfilled', 'MagpieData_mode_NdUnfilled',
				 'MagpieData_minimum_NfUnfilled', 'MagpieData_maximum_NfUnfilled',
				 'MagpieData_range_NfUnfilled', 'MagpieData_mean_NfUnfilled',
				 'MagpieData_avg_dev_NfUnfilled', 'MagpieData_mode_NfUnfilled',
				 'MagpieData_minimum_NUnfilled', 'MagpieData_maximum_NUnfilled',
				 'MagpieData_range_NUnfilled', 'MagpieData_mean_NUnfilled',
				 'MagpieData_avg_dev_NUnfilled', 'MagpieData_mode_NUnfilled',
				 'MagpieData_minimum_GSvolume_pa', 'MagpieData_maximum_GSvolume_pa',
				 'MagpieData_range_GSvolume_pa', 'MagpieData_mean_GSvolume_pa',
				 'MagpieData_avg_dev_GSvolume_pa', 'MagpieData_mode_GSvolume_pa',
				 'MagpieData_minimum_GSbandgap', 'MagpieData_maximum_GSbandgap',
				 'MagpieData_range_GSbandgap', 'MagpieData_mean_GSbandgap',
				 'MagpieData_avg_dev_GSbandgap', 'MagpieData_mode_GSbandgap',
				 'MagpieData_minimum_GSmagmom', 'MagpieData_maximum_GSmagmom',
				 'MagpieData_range_GSmagmom', 'MagpieData_mean_GSmagmom', 'MagpieData_avg_dev_GSmagmom',
				 'MagpieData_mode_GSmagmom', 'MagpieData_minimum_SpaceGroupNumber',
				 'MagpieData_maximum_SpaceGroupNumber', 'MagpieData_range_SpaceGroupNumber',
				 'MagpieData_mean_SpaceGroupNumber', 'MagpieData_avg_dev_SpaceGroupNumber',
				 'MagpieData_mode_SpaceGroupNumber', 'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
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

		self.df.to_csv(r'bandgap_df_new_114.csv', index=False, header=True)


dp = prep()