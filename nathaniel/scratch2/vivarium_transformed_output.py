import pandas as pd
import os

def load_transformed_count_data(directory: str) -> dict:
    """
    Loads each transformed "count space" .hdf output file into a dataframe,
    and returns a dictionary whose keys are the file names and values are
    are the corresponding dataframes.
    """
    dfs = {}
    for entry in os.scandir(directory):
        filename_root, extension = os.path.splitext(entry.name)
        if extension == '.hdf':
#             print(filename_root, type(filename_root), extension, entry.path)
            dfs[filename_root] = pd.read_hdf(entry.path)
    return dfs

def load_count_data_by_location(locations_paths: dict, subdirectory='count_data') -> dict:
    """
    Loads data from all locations into a dictionary of dictionaries of dataframes,
    indexed by location. Each dictionary in the outer dictionary is
    indexed by filename
    
    For each location, reads data files from a directory called f'{path}/{subdirectory}/',
    where `path` is the path for the location specified in the `locations_paths` dictionary.
    """
    locations_count_data = {location: load_transformed_count_data(os.path.join(path, subdirectory))
                        for location, path in locations_paths.items()}
    return locations_count_data

def merge_location_count_data(locations_count_data: dict, copy=True) -> dict:
    """
    Concatenate the count data tables from all locations into a single dictionary of dataframes
    indexed by table name, with a column added to the begininning of each table specifying the
    location for each row of data.
    """
    if copy:
        # Use a temporary variable and a for loop instead of a dictionary comprehension
        # so we can access the `data` variable later.
        locations_count_data_copy = {}
        for location, data in locations_count_data.items():
            locations_count_data_copy[location] = {
                table_name:
                # Use DataFrame.reindex() to simultaneously make a copy of the dataframe and
                # assign a new location column at the beginning.
                table.reindex(columns=['location', *table.columns], fill_value=location)
                for table_name, table in  data.items()
            }
        locations_count_data = locations_count_data_copy
#         # Alternate version using dictionary comprehension; this version would require a different
#         # method of iterating through the table names below, because `data` is inaccessible afterwards.
#         locations_count_data = {
#             location: {
#                 table_name:
#                 table.reindex(columns=['location', *table.columns], fill_value=location)
#                 for table_name, table in  data.items()
#             }
#             for location, data in locations_count_data.items()
#         }
    else:
        # Modify the dictionaries and dataframes in place
        for location, data in locations_count_data.items():
            for table in data.values():
                # Insert a 'location' column at the beginning of each table
                table.insert(0, 'location', location)

    # `data` now refers to the dictionary of count_data tables for the last location
    # encountred in the above for loop. We will use the keys stored in this dictionary
    # to iterate through all the table names and concatenate the tables across all locations.
    all_data = {table_name: pd.concat([locations_count_data[location][table_name]
                                      for location in locations_count_data], copy=False, sort=False)
                for table_name in data}
    return all_data

def load_and_merge_location_count_data(locations_paths: dict, subdirectory='count_data') -> dict:
    locations_count_data = load_count_data_by_location(locations_paths, subdirectory)
    return merge_location_count_data(locations_count_data, copy=False)
