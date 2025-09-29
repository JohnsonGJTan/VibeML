import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

from ....data_processing.utils import str_to_path

from ...base_plot import BasePlot
from ..univariate_registry import univariate_registry

@univariate_registry.register(
    name='Histogram',
    col_types={'continuous'}
)
class Histogram(BasePlot):

    def __init__(self, df: pd.DataFrame, col_name: str):
        super().__init__()
        
        self.summary_statistic['mean'] = df[col_name].mean()
        self.summary_statistic['var'] = df[col_name].var()
        self.summary_statistic['skew'] = df[col_name].skew()
        self.summary_statistic['kurtosis'] = df[col_name].kurtosis()

        sns.histplot(
            data=df,
            x=col_name,
            ax=self.ax,
            stat='density'
        )
        sns.kdeplot(
            data=df, 
            x=col_name, 
            ax=self.ax, 
            color='red'
        )
        self.ax.set_title(f"Histogram of '{col_name}'")
        
    def toggle_kde(self):
        self.ax.lines[0].set_visible(not self.ax.lines[0].get_visible())