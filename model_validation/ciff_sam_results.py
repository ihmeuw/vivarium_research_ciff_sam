"""
Module providing functions and data structures for working with transformed Vivarium output
from the CIFF SAM model.
"""

import collections
import pandas as pd
from vivarium_transformed_output import VivariumTransformedOutput

models = pd.DataFrame(
    [
        [2.4, 'v2.4_corrected_fertility', '2021_08_03_15_08_32'],
        [2.5, 'v2.5_stunting', '2021_08_05_16_17_12'],
        [3.0, 'v3.0_sq_lns', '2021_08_16_17_54_19'],
        [3.1, 'v3.1_sq_lns_stunting_stratified', '2021_08_24_10_28_32']
    ],
    columns=['model_number', 'model_name', 'run_id']
)