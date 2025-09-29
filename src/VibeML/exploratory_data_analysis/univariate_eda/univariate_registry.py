import pandas as pd
from ...data_processing.schema import DataSchema

class UnivariateRegistry:

    COL_TYPES = {
        'continuous',
        'nominal',
        'ordinal',
    }

    def __init__(self):

        self._analysis= {}
        self._valid_col_types = {}

    def register(self, name: str, col_types: set[str],):

        # Check if name is already registered
        if name in self._analysis:
            raise ValueError(f"'{name}' is already a registered function")

        # Check col_types is valid
        if not (col_types <= self.COL_TYPES):
            raise ValueError("Invalid col_types")

        # Register function and corresponding valid column types
        def decorator(analysis):
            self._analysis[name] = analysis
            self._valid_col_types[name] = col_types
            return analysis

        return decorator
    
    def run(self, name, col_name: str, df: pd.DataFrame):
        
        schema = DataSchema.build(df)
        
        # Check if name is in register
        if name not in self._analysis:
            raise ValueError(f"'{name}' not in registry")

        # Check if col_name is in data and is of correct type
        if col_name not in schema.columns:
            raise ValueError(f"'{col_name}' not in schema")
        if schema.get_type(col_name) not in self._valid_col_types[name]:
            raise ValueError(f"'{col_name}' is not a valid column for '{name}'")
        
        return self._analysis[name](df=df, col_name=col_name)

univariate_registry = UnivariateRegistry()
