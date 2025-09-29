import pandas as pd

from ..schema import DataSchema
from .base_process import BaseProcess
from ..process_registry import process_registry

@process_registry.register(
    name='Drop Column',
    col_types = {'continuous', 'nominal', 'ordinal'}
)
class DropCol(BaseProcess):

    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        
        col_names = params['col_names']

        # Check col_names are in schema
        if not (set(col_names) <= set(schema.columns)):
            raise ValueError("col_names contains columns not in schema")

        output_schema = schema.copy()
        for col_name in col_names:
            output_schema = output_schema._del_col(col_name)

        return output_schema

    @staticmethod
    def transform_data(data:pd.DataFrame, params: dict) -> pd.DataFrame:
        
        col_names = params['col_names']

        # Check if col_name in data
        if not (set(col_names) <= set(data.columns)):
            raise ValueError("col_names contains columns not in data")

        return data.drop(col_names, axis=1)

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict):
        return params