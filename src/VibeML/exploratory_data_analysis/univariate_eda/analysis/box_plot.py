import seaborn as sns
import pandas as pd

from ....data_processing.utils import str_to_path

from ...base_plot import BasePlot
from ..univariate_registry import univariate_registry

@univariate_registry.register(
    name='Box Plot',
    col_types={'continuous'}
)
class BoxPlot(BasePlot):
    def __init__(self, df: pd.DataFrame, col_name: str):
        super().__init__()
        sns.boxplot(
            data=df,
            x=col_name,
            ax=self.ax
        )

        self.ax.set_title(f"Box plot for '{col_name}'")

        self.summary_statistic = dict()
        self.summary_statistic['Q1'] = df[col_name].quantile(0.25)
        self.summary_statistic['Q2'] = df[col_name].quantile(0.50)
        self.summary_statistic['Q3'] = df[col_name].quantile(0.75)
        self.summary_statistic['IQR'] = self.summary_statistic['Q3'] - self.summary_statistic['Q1']
        self.summary_statistic['min'] = float(df[col_name].min())
        self.summary_statistic['max'] = float(df[col_name].max())