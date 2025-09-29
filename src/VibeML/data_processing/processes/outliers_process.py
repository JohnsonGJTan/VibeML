import copy

import numpy as np
import pandas as pd

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

@process_registry.register(
    name='Outliers',
    col_types = {'continuous'}
)
class OutliersProcess(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:
        params = copy.deepcopy(params)
        
        col_names = params['col_names']
        outlier_levels = params['outlier_levels']
        # Compute bounds
        outlier_bounds = {}
        for col_name in col_names:
            mean, std = data[col_name].mean(), data[col_name].std()
            outlier_bounds[col_name] = [
                (mean - std*sigma, mean + std*sigma) for 
                sigma in outlier_levels[col_name]
            ]
        params['outlier_bounds'] = outlier_bounds
        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        outlier_levels = params['outlier_levels']
        for col_name in params['col_names']:
            schema = schema._append_ord(
                col_name = col_name + '_outlier',
                ordinal_categories = outlier_levels[col_name]
            )
        
        return schema

    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        
        if 'outlier_bounds' in params:
            outlier_bounds = params['outlier_bounds']
        else:
            outlier_bounds = OutliersProcess.fit_params(data, params)['outlier_bounds']

        data = data.copy()
        outlier_levels = params['outlier_levels']
        impute_cols = {}
        outlier_masks = {}
        for col_name in params['col_names']:
            # Clip data
            lb, ub = outlier_bounds[col_name][-1]
            impute_cols[col_name] = data[col_name].clip(lb, ub)
            conditions = [
                (data[col_name] <= bound[0]) | (data[col_name] >= bound[1])
                for bound in reversed(outlier_bounds[col_name])
            ]
            choices = list(reversed(outlier_levels[col_name]))
            outlier_masks[col_name + '_outlier'] = pd.Series(
                data = np.select(conditions, choices, default = 0),
                name = col_name + '_outlier',
                index = data.index
            )
        
        data = data.assign(**impute_cols).assign(**outlier_masks)
        return data