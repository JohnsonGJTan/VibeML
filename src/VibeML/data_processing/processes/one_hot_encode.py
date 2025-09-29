import copy
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

@process_registry.register(
    name='One Hot Encode',
    col_types = {'nominal', 'ordinal'}
)
class OneHotEncode(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:

        params = copy.deepcopy(params)

        one_hot_encode = OneHotEncoder(
            sparse_output=False,
            handle_unknown=params['handle_unknown'],
        ).fit(data[params['col_names']])
        one_hot_encode.set_output(transform='pandas')

        params['encoder'] = one_hot_encode

        return params
    
    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:

        schema = schema.copy()

        for col_name in params['col_names']:
            if schema.get_type(col_name) == 'nominal':
                categories = schema.nominal[col_name]
            elif schema.get_type(col_name) == 'ordinal':
                categories = schema.ordinal[col_name]
            else:
                raise ValueError(f"'{col_name}' is not categorical.")
            for category in categories:
                schema = schema._append_num(f"{col_name}_{category}")
            schema = schema._del_col(col_name)

        return schema
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:

        data = data.copy()
        # get fitted params
        if 'encoder' in params:
            one_hot_encoder= params['encoder']
        else:
            one_hot_encoder = OneHotEncode.fit_params(data, params)['encoder']

        encoded = cast(pd.DataFrame, one_hot_encoder.transform(data[params['col_names']]))
        return pd.concat([data.drop(params['col_names'], axis=1), encoded], axis=1)