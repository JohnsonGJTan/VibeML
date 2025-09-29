from scipy import stats

import pandas as pd
import numpy as np
import seaborn as sns

from ..bivariate_registry import bivariate_registry
from ...base_plot import BasePlot

@bivariate_registry.register(
    name='Scatter plot',
    col_types=({'continuous'},{'continuous'})
)
class ScatterPlot(BasePlot):

    def __init__(self, df: pd.DataFrame, left_col: str, right_col: str):
        super().__init__()
        values = np.vstack([df[left_col], df[right_col]])
        kernel = stats.gaussian_kde(values)(values)
        sns.scatterplot(
            data=df,
            x=left_col,
            y=right_col,
            c=kernel,
            cmap='viridis',
            edgecolor=None,
            ax=self.ax,
        )
        self.ax.set_title(f"Scatter plot for '{left_col}' vs '{right_col}'")

        self.summary_statistic['Pearson (r)'] = df[left_col].corr(df[right_col], method = 'pearson')
        self.summary_statistic['Spearman (rho)'] = df[left_col].corr(df[right_col], method = 'spearman')
        self.summary_statistic['Kendall (tau)'] = df[left_col].corr(df[right_col], method = 'kendall')