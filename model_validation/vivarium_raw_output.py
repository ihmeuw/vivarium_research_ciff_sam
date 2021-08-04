"""
Module providing functions and data structures for loading and storing raw Vivarium output
(i.e. the output.hdf files).
"""

import pandas as pd
import os

def locaction_paths_from_rundates(base_directory: str, locations_rundates: dict) -> dict:
    """
    Returns a dictionary of the form {'location': 'pathname'} from a dictionary of the form
    {'location': 'rundate'} by constructing each pathname as
    pathname = f'{base_directory}/{location.lower()}/{rundate}/'
    """
    return {location: os.path.join(base_directory, location.lower(), rundate)
            for location, rundate in locations_rundates.items()}

def load_output_by_location(locations_paths: dict, output_filename='output.hdf') -> dict:
    """
    Loads output.hdf file from the path for each location in the dictionary locations_paths,
    and returns a dictionary of output DataFrames indexed by location.
    """
    # Read in data from different countries
    locations_outputs = {location: pd.read_hdf(os.path.join(path, output_filename))
                                               for location, path in locations_paths.items()}
    return locations_outputs

def merge_location_outputs(locations_outputs: dict, copy=True) -> pd.DataFrame:
    """
    Concatenate the output DataFrames for all locations stored in locations_outputs into a single DataFrame,
    with a 'location' column added to specify which location each row came from.
    """
    if copy:
        # Use a dictionary comprhension to avoid overwriting the input dictionary.
        # Use DataFrame.assign to avoid overwriting the dataframes in the dictionary.
        # Note that in the statement `output.assign(location=location)`, the first 'location' is
        # the column name, and the second is the `location` variable, which is the
        # value to assign to the 'location' column.
        locations_outputs = {location: output.assign(location=location)
                             for location, output in locations_outputs.items()}
    else:
        # Modify dictionary and dataframes in place
        for location, output in locations_outputs.items():
            output['location'] = location

    # The concatenated dataframe is not intended to be modified, so we don't
    # need to copy the sub-dataframes.
    return pd.concat(locations_outputs.values(), copy=False, sort=False)

def load_and_merge_location_outputs(locations_paths: dict, output_filename='output.hdf') -> pd.DataFrame:
    """
    Loads output.hdf files from the paths for the locations in the dictionary locations_paths into a single,
    concatenated DataFrame, with a 'location' column added to specify which location each row came from.
    """
    locations_outputs = load_output_by_location(locations_paths, output_filename)
    return merge_location_outputs(locations_outputs, copy=False)
