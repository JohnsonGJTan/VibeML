import copy
from typing import cast

import pandas as pd
from sklearn.preprocessing import TargetEncoder

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

@process_registry.register(
    name='Target Encode',
    col_types = {'nominal', 'ordinal'}
)
class TargetEncode(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:
        params = copy.deepcopy(params)

        target_encoder = TargetEncoder(random_state=42).fit(data[params['col_names']], data[params['target']])
        target_encoder.set_output(transform='pandas')
        
        params['encoder'] = target_encoder

        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:

        for col_name in params['col_names']:
            schema = schema._del_col(col_name)._append_num(col_name)    

        return schema

    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        data = data.copy()
        # get fitted params
        if 'encoder' in params:
            target_encoder = params['encoder']
        else:
            target_encoder= TargetEncode.fit_params(data, params)['encoder']

        encoded = cast(pd.DataFrame, target_encoder.transform(data[params['col_names']]))
        return pd.concat([data.drop(params['col_names'], axis=1), encoded], axis=1)