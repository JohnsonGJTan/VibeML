import pandas as pd
import seaborn as sns

from ...base_plot import BasePlot
from ..univariate_registry import univariate_registry

@univariate_registry.register(
    name='Count Plot',
    col_types={'nominal', 'ordinal'}
)
class CountPlot(BasePlot):
    
    def __init__(self, df: pd.DataFrame, col_name: str):
        super().__init__()

        sns.countplot(
            data=df,
            x=col_name,
            ax=self.ax
        )
        self.ax.set_title(f"Distribution of '{col_name}'")

        self.summary_statistic = df[col_name].value_counts(sort=False).to_dict()