from abc import ABC, abstractmethod
from typing import Optional, Self

import pandas as pd

from ..schema import DataSchema, data_schema_validate
from ..process_registry import process_registry

class BaseProcess(ABC):

    def __init__(self):
        
        self.params_ = None
        self.input_schema_ = None
        self.output_schema_ = None

    @staticmethod
    @abstractmethod
    def fit_params(data: pd.DataFrame, params: dict) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def transform_schema(schema: DataSchema, params: dict) -> DataSchema:
        pass

    @staticmethod
    @abstractmethod
    def transform_data(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        pass

    def fit(self, data: pd.DataFrame, params: dict) -> Self:

        # Check if already fitted else fit
        if self.params_ is not None:
            raise AttributeError(f"Process has already been fitted.")
        else:
            self.params_ = self.fit_params(
                data=data,
                params=params
            )

        # Set input/output schema
        self.input_schema_ = DataSchema.build(data)
        self.output_schema_ = self.transform_schema(schema=self.input_schema_, params=params)

        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.params_ is None:
            raise AttributeError("Process needs to be fitted first.")
        else:
            data_schema_validate(data, self.input_schema_)

        return self.transform_data(data, self.params_)

    def fit_transform(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        return self.fit(data, params).transform(data)