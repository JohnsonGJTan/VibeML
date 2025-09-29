import pandas as pd

from .processor import processor
from .schema import data_schema_validate

class DataPipeline:

    def __init__(self, pipeline: list):

        self.pipeline = pipeline
        self._is_fitted = False
        self.pipeline_ = []
        self.schemas_ = []

    def fit(self, data: pd.DataFrame):
        
        if self._is_fitted:
            raise AttributeError(f"Pipeline is already fitted")

        for pipe in self.pipeline:

            process = processor.get_process(pipe['process_name'])
            data = process.fit_transform(
                data=data,
                params=pipe['params']
            )
            self.pipeline_.append(process)
            self.schemas_.append(process.input_schema_)

        self.schemas_.append(self.pipeline_[-1].output_schema_)

        self._is_fitted = True

        return self

    def transform(self, data: pd.DataFrame):
        
        if not self._is_fitted:
            raise AttributeError(f"Pipeline needs to be fitted first using fit")

        # Validates that data and inptut schema are compatible
        data_schema_validate(data, self.schemas_[0])

        for process in self.pipeline_:
            data = process.transform(data)

        return data