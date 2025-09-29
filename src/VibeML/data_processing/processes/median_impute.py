from collections import deque
import copy

import pandas as pd
import numpy as np

from ..schema import DataSchema
from .base_process import BaseProcess
from ..process_registry import process_registry

'''
We should only perform this on nominal or continuous data.
In addition we assume that downstreamed data will have the fill_value,
otherwise there is data drift.
'''

@process_registry.register(
    name = 'Median Impute',
    col_types ={'continuous', 'ordinal'}
)
class MedianImpute(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:

        schema = DataSchema.build(data)

        params = copy.deepcopy(params)        
        params['fill_vals'] = {}
        col_names = params['col_names']
        
        for col_name in col_names:
            if schema.get_type(col_name) == 'ordinal':
                codes = data[col_name].cat.codes
                valid_codes = codes[codes != -1]
                median_code = int(np.median(valid_codes))
                median_category = data[col_name].cat.categories[median_code]
                params['fill_vals'][col_name] = median_category
            elif schema.get_type(col_name) == 'continuous':
                params['fill_vals'][col_name] = data[col_name].median()

        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        return schema
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:

        data = data.copy()
        col_names = params['col_names']

        # Compute fill_vals
        if 'fill_vals' not in params:
            schema = DataSchema.build(data)
            fill_vals = {}
            for col_name in col_names:
                if schema.get_type(col_name) == 'ordinal':
                    codes = data[col_name].cat.codes
                    valid_codes = codes[codes != -1]
                    median_code = int(np.median(valid_codes))
                    median_category = data[col_name].cat.categories[median_code]
                    fill_vals[col_name] = median_category
                elif schema.get_type(col_name) == 'continuous':
                    fill_vals[col_name] = data[col_name].median()
        else:
            fill_vals = params['fill_vals']

        # Impute fill_vals onto data
        for col_name in col_names:
            data[col_name] = data[col_name].fillna(fill_vals[col_name])

        return data