import pandas as pd

from ...data_processing.schema import DataSchema

class BivariateRegistry:
    
    COL_TYPES = {
        'continuous',
        'nominal',
        'ordinal'
    }
    
    def __init__(self):
        
        self._analysis = {}
        self._valid_col_types = {}

    def register(self, name: str, col_types: tuple[set[str],set[str]]):
        
        if name in self._analysis:
            raise ValueError(f"'{name}' is already a registered function")
        
        if not (col_types[0] | col_types[1] <= self.COL_TYPES):
            raise ValueError("Invalid col_types")
        
        def decorator(analysis):
            self._analysis[name] = analysis
            self._valid_col_types[name] = col_types
            return analysis

        return decorator

    def run(self, name, col_names: tuple[str, str], df: pd.DataFrame):
        
        schema = DataSchema.build(df)
        if name not in self._analysis:
            raise ValueError(f"'{name}' not in registry")
        
        if not ({col_name for col_name in col_names} <= set(schema.columns)):
            raise ValueError(f"'{col_names}' not in schema")
        if any([schema.get_type(col_name) not in valid_col_types for col_name, valid_col_types in zip(col_names, self._valid_col_types[name])]):
            raise ValueError(f"Invalid column types")

        return self._analysis[name](df=df, left_col = col_names[0], right_col = col_names[1])


bivariate_registry = BivariateRegistry()