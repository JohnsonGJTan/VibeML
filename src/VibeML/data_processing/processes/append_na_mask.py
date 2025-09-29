import pandas as pd

from ..schema import DataSchema
from .base_process import BaseProcess
from ..process_registry import process_registry

@process_registry.register(
    name='Append Null Mask',
    col_types={'continuous', 'nominal', 'ordinal'}
)
class AppendNullMask(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:
        return params

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        col_names = params['col_names']
        schema_transformed = schema.copy()
        for col_name in col_names:
            schema_transformed = schema_transformed._append_num(col_name + '_null')
        return schema_transformed

    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        
        col_names = params['col_names']

        data = data.copy()
        for col_name in col_names:
            mask = pd.Series(data[col_name].isnull(), name= col_name + '_null')
            data = pd.concat([data, mask], axis=1)
        
        return data