import numpy as np
import pandas as pd
import seaborn as sns

from ...base_plot import BasePlot
from ..univariate_registry import univariate_registry


def outlier_strat(data: pd.Series):

    lr_mask = np.select(
        condlist=[data < data.mean(), data > data.mean()],
        choicelist = ['left', 'right'],
        default='center'
    )

    outlier_levels = (data - data.mean()).abs().floordiv(data.std()).astype(int)

    return pd.DataFrame({
        'level': outlier_levels,
        'mask': lr_mask
    })

@univariate_registry.register(
    name='Outlier Plot',
    col_types={'continuous'}
)
class OutlierPlot(BasePlot):

    def __init__(self, df: pd.DataFrame, col_name: str):
        super().__init__()
        outlier_df = outlier_strat(df[col_name])

        sns.histplot(outlier_df, x='level', discrete=True, hue='mask', multiple='stack', ax=self.ax)
        self.ax.set_title(f"Outlier plot of '{col_name}'")
        self.ax.set_xlabel("Outlier level (deviation from mean)")

        for level, mask in pd.crosstab(outlier_df['mask'], outlier_df['level']).to_dict().items():
            self.summary_statistic[level] = f"{mask['left'] + mask['right']} ({mask['left']}/{mask['right']})"