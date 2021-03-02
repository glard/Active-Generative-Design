# This file is used to plot trend of optimization process.
import seaborn as sns
import numpy as np
import pandas as pd

class TrendPlot:
    @staticmethod
    def bst_vs_no(bst_list):
        y = np.array(bst_list)
        x = np.arange(1, len(bst_list) + 1, 1)
        y = pd.Series(y)
        x = pd.Series(x)
        plot = sns.lineplot(x=x, y=y, sort=False, label="Best Value found")
        plot.set(ylabel='band gap')
        plot.set(xlabel='Probe times')
        plot.figure.savefig("_" + str(len(bst_list)) + "AGD_process.pdf")