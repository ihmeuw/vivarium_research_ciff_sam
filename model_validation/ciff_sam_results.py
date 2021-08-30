"""
Module providing functions and data structures for working with transformed Vivarium output
from the CIFF SAM model.
"""

import collections
import pandas as pd
from model_validation.vivarium_transformed_output import VivariumTransformedOutput

class VivariumMeasures(VivariumTransformedOutput, collections.abc.MutableMapping):
    """Implementation of the MutableMapping abstract base class to conveniently store transformed
    Vivarium count data tables as object attributes and to store and manipulate additional tables
    computed from the raw data.

    See https://docs.python.org/3/library/collections.abc.html

    Example usage:
    --------------
    count_data_path = (
        '/ihme/costeffectiveness/results/vivarium_ciff_sam/'
        'v2.3_wasting_birth_prevalence/ciff_sam/2021_07_26_17_14_31/count_data'
    )
    orig_data = VivariumTransformedOutput.from_directory(count_data_path) # Implements Mapping but not MutableMapping
    data = VivariumMeasures(orig_data) # Copies the original data to an object implementing MutableMapping
    data.table_names() # Displays available count_data tables for this model run
    # Access the tables via object attributes:
    data.deaths # deaths data table
    data.wasting_state_person_time # wasting state person time table
    """
    @classmethod
    def from_model_spec(cls, model_id, run_id=None):
        """Create a VivariumMeasures object from the model_id (e.g. 1.0, 1.1, 2.0, etc.) and optionally run_id
        (i.e. the folder name of the form 'yyyy_mm_dd_hh_mm_ss' indicating when the run was launched), using the
        `get_count_data_path` function to get the path to the count_data for the specified model.

        Example usage:
        --------------
        # This is a shortcut to creating an object equivalent to the one in the example from the class description
        data = VivariumMeasures.from_model_spec(2.3) # run_id can be omitted if there is only one run_id for the model
        # data = VivariumMeasures.from_model_spec(2.3, '2021_07_26_17_14_31') # does the same thing
        data.table_names() # Displays available count_data tables for this model run
        """
        return cls.from_directory(get_count_data_path(model_id, run_id))

    @classmethod
    def cleaned_from_model_spec(cls, model_id, run_id=None):
        """Create a VivariumMeasures object from the model_id (e.g. 1.0, 1.1, 2.0, etc.) and optionally run_id
        (i.e. the folder name of the form 'yyyy_mm_dd_hh_mm_ss' indicating when the run was launched), with
        the data tables reformatted using the `clean_transformed_data` function.

        Example usage:
        --------------
        data = VivariumMeasures.cleaned_from_model_spec(2.3) # run_id can be omitted if there is only one run_id for the model
        # data = VivariumMeasures.cleaned_from_model_spec(2.3, '2021_07_26_17_14_31') # does the same thing
        data.table_names() # Displays available count_data tables for this model run
        """
        return cls(clean_transformed_data(cls.from_model_spec(model_id, run_id)))

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        del self.key

project_results_directory = '/ihme/costeffectiveness/results/vivarium_ciff_sam'

models = pd.DataFrame(
    [
        [2.3, 'v2.3_wasting_birth_prevalence', '2021_07_26_17_14_31'],
        [2.4, 'v2.4_corrected_fertility', '2021_08_03_15_08_32'],
        [2.5, 'v2.5_stunting', '2021_08_05_16_17_12'],
        [3.0, 'v3.0_sq_lns', '2021_08_16_17_54_19'],
        [3.1, 'v3.1_sq_lns_stunting_stratified', '2021_08_24_10_28_32']
    ],
    columns=['model_id', 'model_name', 'run_id']
)

def get_count_data_path(model_id, run_id=None, models_df=models):
    """Returns the path to the count_data folder for the specified model number and run_id (timestamp string).
    If there is only one run_id in `models_df` corresponding to the requested model_id, then run_id
    does not need to be specified.
    """
    model_metadata = models_df.set_index('model_id').loc[model_id]
    if run_id is None:
        if len(model_metadata[['run_id']]) > 1:
            raise ValueError(
                f"You must specify the run_id for model_id {model_id} because"
                f" there is more than one run_id:\n{model_metadata}"
            )
        model_name, run_id = model_metadata.loc[['model_name', 'run_id']]
    else:
        matching_run = model_metadata['run_id']==run_id
        if isinstance(matching_run, bool) and matching_run:
            # There was only one row corresponding to this model, and model_metadata is a Series
            pass
        elif isinstance(matching_run, pd.Series) and len(model_metadata.loc[matching_run]) > 0:
            # There was more than one row, so model_metadata is a DataFrame and matching_run is a Series
            # Filter to metadata for the requested run_id and convert to a Series:
                model_metadata = model_metadata.loc[matching_run].squeeze()
        else:
            raise ValueError(f"No run_id {run_id} for model_id {model_id}:\n{model_metadata}")
        model_name = model_metadata['model_name']
#         print(model_metadata, len(model_metadata), model_name,'\n',run_id)
    model_count_data_path = f'{project_results_directory}/{model_name}/ciff_sam/{run_id}/count_data/'
    return model_count_data_path

def clean_transformed_data(data):
    """Reformat transformed count data to make more sense.

    Parameters
    ----------
    data: Mapping of table names to DataFrames
        The transformed data tables to clean, from the CIFF SAM model.
    """
    # Create a VivariumMeasures object with the same tables stored in `data`
    clean_data = VivariumMeasures(data)
    # Define a function to make the transition count dataframes better
    def clean_transition_df(df):
        return (df
                .assign(transition=lambda df: df['measure'].str.replace('_event_count', ''))
                .assign(measure='transition_count')
               )
    # Make the wasting and disease transition count dataframes better
    clean_data.update(
        {table_name: clean_transition_df(table) for table_name, table in data.items()
         if table_name.endswith('transition_count')}
    )
    if 'wasting_state_person_time' in data:
        # Rename mislabeled 'cause' column in `wasting_state_person_time`
        clean_data['wasting_state_person_time'] = (
            data['wasting_state_person_time'].rename(columns={'cause':'wasting_state'})
        )
    if 'disease_state_person_time' in data:
        # Rename poorly named 'cause' column in `disease_state_person_time` and add an actual cause column
        clean_data['disease_state_person_time'] = (
            data['disease_state_person_time']
            .rename(columns={'cause':'cause_state'})
            .assign(cause=lambda df: df['cause_state'].str.replace('susceptible_to_', ''))
        )
    return clean_data