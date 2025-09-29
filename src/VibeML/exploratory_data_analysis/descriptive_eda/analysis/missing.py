from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import numpy as np

from ....data_processing.utils import str_to_path
from ..descriptive_registry import descriptive_registry

@descriptive_registry.register(name='Missing analysis')
class missing_analysis:

    def __init__(self, df: pd.DataFrame):
        
        self.fig, self.ax = plt.subplots(1, 2, layout='constrained')
        self.fig.suptitle("Missing analysis")

        # plot matrix
        msno.matrix(df=df, ax=self.ax[0], sparkline=False, fontsize=8)
        # plot heatmap
        msno.heatmap(df=df, ax=self.ax[1], fontsize=8)

        self.summary_statistic = {}
        missing_count = df.isnull().sum().to_dict()
        for col, count in missing_count.items():
            self.summary_statistic[col] = (count, count/len(df))
        

    def save(self, path: str| Path):
        path = str_to_path(path)
        self.fig.savefig(path)