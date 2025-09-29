from collections import deque

import pandas as pd
import copy

from ..schema import DataSchema
from .base_process import BaseProcess
from ..process_registry import process_registry

'''
There is an assumption that downstreamed data will have the fill_value,
otherwise there is data drift.
In addition we should only perform this on nominal data
'''

@process_registry.register(
    name='Mode Impute',
    col_types = {'nominal'}
)
class ModeImpute(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:

        params = copy.deepcopy(params)        
        col_names = params['col_names']
        params['fill_vals'] = {col_name: data[col_name].mode()[0] for col_name in col_names}

        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        return schema
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:

        data = data.copy()
        col_names = params['col_names']

        if 'fill_vals' not in params:
            fill_vals = {col_name: data[col_name].mode()[0] for col_name in col_names}
        else:
            fill_vals = params['fill_vals']

        # Impute fill_vals onto data
        for col_name in col_names:
            data[col_name] = data[col_name].fillna(fill_vals[col_name])

        return data