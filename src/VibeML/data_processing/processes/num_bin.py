import copy 

import pandas as pd
import numpy as np

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

def bin_to_categories(bins: list[int]):
    bin_categories = [f"(-inf, {bins[0]}]"]
    for i in range(len(bins) - 1):
        bin_categories.append(f"({bins[i]}, {bins[i+1]}]")
    bin_categories.append(f"({bins[-1]}, inf)")

    return bin_categories

@process_registry.register(
    name='Numerical Bin',
    col_types = {'continuous'}
)
class NumericalBin(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:
        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        for col_name in params['col_names']:
            bins = params['bins'][col_name]
            bin_categories = bin_to_categories(bins)
            schema = schema._del_col(col_name)._append_ord(col_name, bin_categories)
        return schema
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        for col_name in params['col_names']:
            bins = params['bins'][col_name]
            binned_col = pd.cut(
                x=data[col_name],
                bins= [-np.inf] + bins + [np.inf],
                labels=bin_to_categories(bins),
                ordered=True
            )
            data = data.assign(**{col_name: binned_col})
        return data