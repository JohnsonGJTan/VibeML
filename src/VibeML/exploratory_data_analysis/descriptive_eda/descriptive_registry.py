import pandas as pd

class DescriptiveRegistry:

    def __init__(self):

        self._analysis = {}

    def register(self, name: str):

        if name in self._analysis:
            raise ValueError(f"'{name}' is already a registered function")
            
        def decorator(analysis):
            self._analysis[name] = analysis
            return analysis
        
        return decorator
    
    def run(self, name: str, df: pd.DataFrame):

        if name not in self._analysis:
            raise ValueError(f"'{name}' not in registry")
        return self._analysis[name](df=df)


descriptive_registry = DescriptiveRegistry()