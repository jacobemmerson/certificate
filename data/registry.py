'''
author: @tae
'''

import argparse
from data.benchmarks.base import REGISTRY

# ----- Do Not Modify -----
AVAILABLE_DATASETS = list(REGISTRY.keys())

# ----- Parser ------

def parse():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    choices = AVAILABLE_DATASETS + ["all"]
    parser.add_argument(
        "-d",
        "--dataset",
        default="all",
        choices=choices,
        help=f"Path to the specified RAW dataset to be formatted for inspect. Must be one of {choices}"
    )

    dataset = parser.parse_args().dataset
    return AVAILABLE_DATASETS if dataset == "all" else [dataset]

# ----- Main -----
if __name__ == "__main__":
    
    # Debug
    DATASETS = parse() # -> List[one of or all AVAILABLE_DATASETS]
    for file in DATASETS:
        dataset = REGISTRY["socialharmbench"]()
        dataset._debug()