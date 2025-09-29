class ProcessRegistry:

    def __init__(self):
        
        self._processes = {}
        self._col_types = {}

    def register(self, name: str, col_types: set[str]):
        
        if name in self._processes:
            raise ValueError(f"'{name}' is already a registered function.")

        def decorator(process):
            self._processes[name] = process
            self._col_types[name] = col_types
            return process

        return decorator
    
    def get_process(self, name: str):
        
        if name not in self._processes:
            raise ValueError(f"'{name}' is not a registered process.")
        
        return self._processes[name]()


process_registry = ProcessRegistry()