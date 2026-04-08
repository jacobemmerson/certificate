'''
This contains the abstract class that defines how datasets should be initialized for the codebase. 
Each dataset may need to be formatted differently based on the public/private set as well as the datasets properties relative to other benchmarks
Following this practice will help maintain its readability and extensibility when new datasets are added.

author: @tae
'''

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

# ----- Helpers ------
RAW_DATA_DIR = Path(__file__).parent / "raw"
REGISTRY = {} # this is what is accessed in format.py

def get_path(file, private):
    if private:
        # TODO: Add check if the private directory is empty
        return RAW_DATA_DIR / "private" / f"{file}.csv"
    else:
        return RAW_DATA_DIR / "public" / f"{file}.csv"

# ----- Init Registry -----
class Benchmark(ABC):

    name = None # must match the name of the CSV in public/raw
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):

            assert isinstance(cls.name, str), "You must provide a STRING name for the dataset."
            REGISTRY[cls.name] = cls

    def load_dataset(self, file, private=False):
        '''
        Loads the provided dataset; if the raw dataset directory changes (the one with the original CSVs), modify RAW_DATA_DIR
        '''
        return pd.read_csv(get_path(file, private))

    @abstractmethod
    def format(self) -> pd.DataFrame:
        pass
