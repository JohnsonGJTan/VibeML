import seaborn as sns
import pandas as pd

from ...base_plot import BasePlot
from ..bivariate_registry import bivariate_registry

@bivariate_registry.register(
    name='Box plot',
    col_types= (
        {'continuous'},
        {'nominal','ordinal'}
    )
)
class BivariateBoxPlot(BasePlot):

    def __init__(self, df: pd.DataFrame, left_col: str, right_col: str):
        super().__init__()
        sns.boxplot(
            data=df,
            x=right_col,
            y=left_col,
            #hue=right_col,
            ax=self.ax,
        )
        self.ax.set_title(f"Box plot for '{left_col}' grouped by '{right_col}'")
        # TODO: Summary statistics