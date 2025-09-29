import copy 

import pandas as pd

from .base_process import BaseProcess
from ..schema import DataSchema
from ..process_registry import process_registry

@process_registry.register(
    name='Categorical Group',
    col_types = {'nominal'}
)
class CategoricalGroup(BaseProcess):

    @staticmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:
        return params
    
    @staticmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        
        col_names = params['col_names']
        maps = params['maps']

        for col_name in col_names:
            map_categories = list(set(maps[col_name].values()))
            schema._del_col(col_name)._append_unord(col_name, map_categories)

        return schema
    
    @staticmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        
        for col_name in params['col_names']:
            col_grouped = pd.Series(pd.Categorical(
                values = data[col_name].map(params['maps'][col_name]),
                categories = list(set(params['maps'][col_name].values())),
                ordered = False
            ))
            data = data.assign(**{col_name: col_grouped})

        return data