import pandas as pd
import seaborn as sns

from ...base_plot import BasePlot
from ..bivariate_registry import bivariate_registry

@bivariate_registry.register(
    name='Histogram',
    col_types=(
        {'continuous'},
        {'nominal', 'ordinal'}
    )
)
class Histogram(BasePlot):

    def __init__(self, df: pd.DataFrame, left_col: str, right_col: str):
        super().__init__()
        sns.histplot(
            data=df,
            x=left_col,
            hue=right_col,
            stat='density',
            multiple='dodge',
            ax=self.ax,
        )
        self.ax.set_title(f"Histogram of '{left_col}'")

        kde = sns.kdeplot(
            data=df,
            x=left_col,
            hue=right_col,
            ax=self.ax,
        )

    def toggle_kde(self):
        for line in self.ax.lines:
            line.set_visible(not line.get_visible())