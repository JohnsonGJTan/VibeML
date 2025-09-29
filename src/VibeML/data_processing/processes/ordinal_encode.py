import copy
from typing import cast

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

@process_registry.register(
    name='Ordinal Encode',
    col_types={'ordinal'}
)
class OrdinalEncode(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:

        params = copy.deepcopy(params)

        ordinal_encoder = OrdinalEncoder(
            categories=params['orders'],
            handle_unknown=params['handle_unknown'],
            unknown_value=params['unknown_value']
        )
        ordinal_encoder.set_output(transform='pandas')
        ordinal_encoder.fit(data[params['col_names']])

        params['encoder'] = ordinal_encoder
        
        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        
        for col_name in params['col_names']:
            schema = schema._del_col(col_name)._append_num(col_name)

        return schema
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:

        # get fitted params
        if 'encoder' in params:
            ordinal_encoder = params['encoder']
        else:
            ordinal_encoder = OrdinalEncode.fit_params(data, params)['encoder']

        encoded = cast(pd.DataFrame, ordinal_encoder.transform(data[params['col_names']]))
        return pd.concat([data.drop(params['col_names'], axis=1), encoded], axis=1)