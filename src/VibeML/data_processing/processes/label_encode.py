import copy
from typing import cast

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

@process_registry.register(
    name='Label Encode',
    col_types = {'nominal'}
)
class LabelEncode(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:

        params = copy.deepcopy(params)

        label_encoder = LabelEncoder().fit(data[params['col_name']])
        params['encoder'] = label_encoder

        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        col_name = params['col_name']
        return schema._del_col(col_name)._append_num(col_name)
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:

        if 'encoder' in params:
            label_encoder = params['encoder']
        else:
            label_encoder = LabelEncode.fit_params(data, params)['encoder']

        encoded = cast(np.ndarray, label_encoder.transform(data[params['col_name']]))
        return data.assign(**{params['col_name']: encoded.tolist()})