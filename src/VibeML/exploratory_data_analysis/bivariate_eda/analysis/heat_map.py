import pandas as pd
import seaborn as sns

from ...base_plot import BasePlot
from ..bivariate_registry import bivariate_registry

@bivariate_registry.register(
    name='Heat map',
    col_types = (
        {'nominal', 'ordinal'},
        {'nominal', 'ordinal'}
    )
)
class HeatMap(BasePlot):

    def __init__(self, df: pd.DataFrame, left_col: str, right_col: str):
        super().__init__()
        
        sns.heatmap(
            data=pd.crosstab(df[left_col], df[right_col]),
            ax=self.ax,    
        )
        self.ax.set_title(f"Heat map for '{left_col}' vs '{right_col}'")