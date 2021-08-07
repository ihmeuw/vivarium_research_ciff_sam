import pandas as pd

VALUE_COLUMN = 'value'
DRAW_COLUMN  = 'input_draw'
SCENARIO_COLUMN = 'scenario'

INDEX_COLUMNS = [DRAW_COLUMN, SCENARIO_COLUMN]

def set_global_index_columns(index_columns:list)->None:
    """
    Set INDEX_COLUMNS to a custom list of columns for the Vivarium model output.
    For example, if tables for different locations have been concatenated with
    a new column called 'location', then use the following to get the correct
    behavior for the functions in this module:
    
    set_global_index_columns(['location']+lsff_output_processing.INDEX_COLUMNS)
    """
    global INDEX_COLUMNS
    INDEX_COLUMNS = index_columns

def ratio(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    strata: list,
    multiplier=1,
    numerator_broadcast=None,
    dropna=False,
)-> pd.DataFrame: # or just numerator_broadcast instead of having two separate arguments
    """
    Compute a ratio or rate by dividing the numerator by the denominator.

    Parameters
    ----------

    numerator : DataFrame
        The numerator data for the ratio or rate.

    denominator : DataFrame
        The denominator data for the ratio or rate.

    strata : list of column names present in the numerator and denominator
        The stratification variables for the ratio or rate.

    multiplier : int or float, default 1
        Multiplier for the numerator, typically a power of 10,
        to adjust the units of the result. For example, if computing a ratio,
        some multipliers with corresponding units are:
        1 - proportion
        100 - percent
        1000 - per thousand
        100_000 - per hundred thousand

    numerator_broadcast : list of column names present in the numerator, or None
        Additional columns in the numerator by which to stratify or broadcast.
        Note that the population in the numerator must always be a subset of
        the population in the denominator, so it only makes sense to include
        addiional strata in the numerator.

        For example, if 'sex' is included in `numerator_broadcast` but not `strata`,
        then the resulting ratio can be interpreted as a joint distribution over sex,
        and summing over the 'sex' column in the ratio would give the same result as
        passing numerator_broadcast=None.

        You can also pass columns to `numerator_broadcast` to do muliple computations
        at the same time. E.g. pass 'cause' to compute a ratio or rate for multiple causes
        at once, or pass 'measure' to compute a ratio or rate for multiple measures at
        once (like deaths, ylls, and ylds).

    dropna : boolean, default False
         Whether to drop rows with NaN values in the result, namely
         if division by 0 occurs because of an empty stratum in the denominator.

     Returns
     -------
     ratio : DataFrame
         The ratio or rate data = numerator / denominator.
    """
    index_cols = INDEX_COLUMNS

    if numerator_broadcast is None:
        numerator_broadcast = []

    numerator = numerator.groupby(strata+index_cols+numerator_broadcast)[VALUE_COLUMN].sum()
    denominator = denominator.groupby(strata+index_cols)[VALUE_COLUMN].sum()

    ratio = (numerator / denominator) * multiplier

    # If dropna is True, drop rows where we divided by 0
    if dropna:
        ratio.dropna(inplace=True)

    return ratio.reset_index()

def averted(measure, baseline_scenario, scenario_col=None):
    """
    Compute an "averted" measure (e.g. DALYs) or measures by subtracting
    the intervention value from the baseline value.
    
    Parameters
    ----------
    
    measure : DataFrame
        DataFrame containing both the baseline and intervention data.
        
    baseline_scenario : scalar, typically str
        The name or other identifier for the baseline scenario in the
        `scenario_col` column of the `measure` DataFrame.
        
    scenario_col : str, default None
        The name of the scenario column in the `measure` DataFrame.
        Defaults to the global parameter SCENARIO_COLUMN if None is passed.
        
    Returns
    -------
    
    averted : DataFrame
        The averted measure(s) = baseline - intervention
    """
    
    scenario_col = SCENARIO_COLUMN if scenario_col is None else scenario_col
    
    # Filter to create separate dataframes for baseline and intervention
    baseline = measure[measure[scenario_col] == baseline_scenario]
    intervention = measure[measure[scenario_col] != baseline_scenario]
    
    # Columns to match when subtracting intervention from baseline
    index_columns = sorted(set(baseline.columns) - set([scenario_col, VALUE_COLUMN]),
                           key=baseline.columns.get_loc)
    print(index_columns)
    
    # Put the scenario column in the index of intervention but not baseline.
    # When we subtract, this will broadcast over different interventions if there are more than one.
    baseline = baseline.set_index(index_columns)
    intervention = intervention.set_index(index_columns+[scenario_col])
    print('baseline index:', baseline.index.names)
    print('intervention index:', intervention.index.names)
    
    # Get the averted values
    averted = baseline[[VALUE_COLUMN]] - intervention[[VALUE_COLUMN]]
    print('averted index:', averted.index.names)
    
    # Insert a column after the scenario column to record what the baseline scenario was
    averted = averted.reset_index()
    print(averted.columns)
    averted.insert(averted.columns.get_loc(scenario_col)+1, 'relative_to', baseline_scenario)
    
    return averted

def difference(measure:pd.DataFrame, identifier_col:str, minuend_id=None, subtrahend_id=None)->pd.DataFrame:
    """
    Returns the difference of a measure stored in the measure DataFrame, where the
    rows for the minuend (that which is diminished) and subtrahend (that which is subtracted)
    are determined by the values in identifier_col
    """
    if minuend_id is not None:
        minuend = measure[measure[identifier_col] == minuend_id]
        if subtrahend_id is not None:
            subtrahend = measure[measure[identifier_col] == subtrahend_id]
        else:
            # Use all values not equal to minuend_id for subtrahend (minuend will be broadcast over subtrahend)
            subtrahend = measure[measure[identifier_col] != minuend_id]
    elif subtrahend_id is not None:
        subtrahend = measure[measure[identifier_col] == subtrahend_id]
        # Use all values not equal to subtrahend_id for minuend (subtrahend will be broadcast over minuend)
        minuend = measure[measure[identifier_col] != subtrahend_id]
    else:
        raise ValueError("At least one of `minuend_id` and `subtrahend_id` must be specified")

    # Columns to match when subtracting subtrahend from minuend
    # Oh, I just noticed that I could use the Index.difference() method here, which I was unaware of before...
    index_columns = sorted(set(measure.columns) - set([identifier_col, VALUE_COLUMN]),
                           key=measure.columns.get_loc)

    minuend = minuend.set_index(index_columns)
    subtrahend = subtrahend.set_index(index_columns)

    # Add the identifier column to the index of the larger dataframe
    # (or default to the subtrahend dataframe if neither needs broadcasting).
    if minuend_id is None:
        minuend.set_index(identifier_col, append=True)
    else:
        subtrahend.set_index(identifier_col, append=True)

    # Subtract DataFrames, not Series, because Series will drop the identifier column from the index
    # if there is no broadcasting. (Behavior for Series and DataFrames is different - is this a
    # feature or a bug in pandas?)
    difference = minuend[[VALUE_COLUMN]] - subtrahend[[VALUE_COLUMN]]
    difference = difference.reset_index()

    # Add a column to specify what was subtracted from (the minuend) or what was subtracted (the subtrahend)
    colname, value = 'subtracted_from', minuend_id if minuend_id is not None else 'subtracted_value', subtrahend_id
    difference.insert(difference.columns.get_loc(identifier_col)+1, colname, value)

    return difference

def describe(data, **describe_kwargs):
    """Wrapper function for DataFrame.describe() with `data` grouped by everything except draw and value."""
    groupby_cols = [col for col in data.columns if col not in [DRAW_COLUMN, VALUE_COLUMN]]
    return data.groupby(groupby_cols)[VALUE_COLUMN].describe(**describe_kwargs)

def get_mean_lower_upper(described_data, colname_mapper={'mean':'mean', '2.5%':'lower', '97.5%':'upper'}):
    """
    Gets the mean, lower, and upper value from `described_data` DataFrame, which is assumed to have
    the format resulting from a call to DataFrame.describe().
    """
    return described_data[colname_mapper.keys()].rename(columns=colname_mapper).reset_index()
