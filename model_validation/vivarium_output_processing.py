import pandas as pd
import collections

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

def _listify_singleton_cols(colnames, df):
    """Wrap a single column name in a list, or return colnames unaltered if it's already a list of column names."""

    def method1(colnames, df):
        """Method 1 (doesn't depend on df): Assume that if colnames has a type that is in a whitelist of
        allowed iterable types, then it is an iterable of column names, and otherwise it must be a single
        column name.
        """
        if not isinstance(colnames, (list, pd.Index)):
            colnames = [colnames]
        return colnames

    def method2(colnames, df):
        """Method 2: Assume that if colnames is hashable it represents a single column name,
        and otherwise it must be an iterable of column names. (This method doesn't allow tuples of column
        names since tuples are hashable.)
        """
        if isinstance(colnames, collections.Hashable):
            # This line could still raise an 'unhashable type' TypeError if e.g. colnames is a tuple
            # that contains an unhashable type
            if colnames in df: # assume colnames is a single column name in df
                colnames = [colnames]
            else: # Assume colnames is supposed to be a single column name
                raise KeyError(f"Key {colnames} not in the DataFrame")
        elif not isinstance(colnames, collections.Iterable): # assume colname is an iterable of column names
            raise ValueError(f"{colnames} must be a single column name in df or an iterable of column names")
        return colnames

    def method3(colnames, df):
        """Method 3: Assume that if colnames is a string or is a hashable object that is in the dataframe's columns
        (e.g. a tuple), then it represents a single column namee. Otherwise it must be an iterable of column names.
        (This method allows tuples of column names.)
        """
        if isinstance(colnames, collections.Hashable):
            # This line could still raise an 'unhashable type' TypeError if e.g. colnames is a tuple
            # that contains an unhashable type
            if colnames in df: # assume colnames is a single column name in df
                colnames = [colnames]
            elif isinstance(colnames, str): # Assume colnames is supposed to be a single column name
                raise KeyError(f"string {colnames} not in the DataFrame")
        elif not isinstance(colnames, collections.Iterable): # assume colname is an iterable of column names
            raise ValueError(f"{colnames} must be a single column name in df or an iterable of column names")
        return colnames

    return method1(colnames, df) # Go with the most restrictive method for now

def marginalize(df:pd.DataFrame, marginalized_cols, value_cols=VALUE_COLUMN, reset_index=True)->pd.DataFrame:
    """Sum the values of a dataframe over the specified columns to marginalize out.

    https://en.wikipedia.org/wiki/Marginal_distribution

    The `marginalize` and `stratify` functions are complementary in that the two functions do the same thing
    (sum values of a dataframe over a subset of the dataframe's columns), but the specified columns
    in the second argument of the two functions are opposites:
        For `marginalize` you specify the marginalized columns you want to sum over, whereas
        for `stratify` you specify the stratification columns that you want to keep un-summed.

    Parameters
    ----------

    df: DataFrame
        A dataframe with at least one "value" column to be aggregated, and additional "identifier" columns,
        at least one of which is to be marginalized out. That is, the data in the "value" column(s) will be summed
        over all catgories in the "marginalized" column(s). All columns in the dataframe are assumed to be either
        "value" columns or "identifier" columns, and the columns to marginalize should be a subset of the
        identifier columns.

    martinalized_cols: single column label, list of column labels, or pd.Index object
        The column(s) to sum over (i.e. marginalize)

    value_cols: single column label, list of column labels, or pd.Index object
        The column(s) in the dataframe that contain the values to sum

    reset_index: bool
        Whether to reset the dataframe's index after calling groupby().sum()

    Returns
    ------------
    summed_data: DataFrame
        DataFrame with the summed values, whose columns are the same as those in df except without `marginalized_cols`,
        which have been aggregated over.
        If reset_index == False, all the resulting columns will be placed in the DataFrame's index except for `value_cols`.
    """
    marginalized_cols = _listify_singleton_cols(marginalized_cols, df)
    value_cols = _listify_singleton_cols(value_cols, df)
    # Move Index levels into columns to enable passing index level names as well as column names to marginalize
    if df.index.nlevels > 1 or df.index.name in marginalized_cols:
        df = df.reset_index()
    index_cols = df.columns.difference([*marginalized_cols, *value_cols]).to_list()
    summed_data = df.groupby(index_cols, observed=True)[value_cols].sum() # observed=True needed for Categorical data
    return summed_data.reset_index() if reset_index else summed_data

def stratify(df: pd.DataFrame, strata, value_cols=VALUE_COLUMN, reset_index=True)->pd.DataFrame:
    """Sum the values of the dataframe so that the reult is stratified by the specified strata.

    https://en.wikipedia.org/wiki/Stratification_(clinical_trials)

    More specifically, `stratify` groups `df` by the stratification columns and sums the value columns,
    but automatically adds INDEX_COLS (usually DRAW_COLUMN and SCENARIO_COLUMN) to the `by` parameter
    of the groupby. That is, the return value is df.groupby(strata+INDEX_COLS)[value_cols].sum()

    The `marginalize` and `stratify` functions are complementary in that the two functions do the same thing
    (sum values of a dataframe over a subset of the dataframe's columns),but the specified columns
    in the second argument of the two functions are opposites:
        For `marginalize` you specify the marginalized columns you want to sum over, whereas
        for `stratify` you specify the stratification columns that you want to keep un-summed.

    Parameters
    ----------

    df: DataFrame
        A dataframe with at least one "value" column to be aggregated, and additional "identifier" columns
        which must include those listed in INDEX_COLS and potentially other columns to stratify by.
        That is, the data in the "value" column(s) will be summed over all catgories in the identifier
        column except those in `strata` and INDEX_COLS. All columns in the dataframe are assumed to be either
        "value" columns or "identifier" columns, and the columns to stratify by should be a subset of the
        identifier columns.

    strata: single column label, list of column labels, or pd.Index object
        The column(s) to stratify by (i.e. group by before summing)

    value_cols: single column label, list of column labels, or pd.Index object
        The column(s) in the dataframe that contain the values to sum

    reset_index: bool
        Whether to reset the dataframe's index after calling groupby().sum()

    Returns
    ------------

    summed_data: DataFrame
        DataFrame with the summed values, whose columns are the columns listed in `strata` and INDEX_COLS
        (usually DRAW_COLUMN and SCENARIO_COLUMN), with all other columns being marginalized out.
        If reset_index == False, all the resulting columns will be placed in the DataFrame's index except
        for `value_cols`.
    """
    strata = _listify_singleton_cols(strata, df)
    value_cols = _listify_singleton_cols(value_cols, df)
    index_cols = [*strata, *INDEX_COLUMNS]
    summed_data = df.groupby(index_cols, observed=True)[value_cols].sum()
    return summed_data.reset_index() if reset_index else summed_data

def ratio(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    strata: list,
    multiplier=1,
    numerator_broadcast=None,
    denominator_broadcast=None,
    dropna=False,
    reset_index=True,
)-> pd.DataFrame:
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
    else:
        numerator_broadcast = _listify_singleton_cols(numerator_broadcast, numerator)

    if denominator_broadcast is None:
        denominator_broadcast = []
    else:
        denominator_broadcast = _listify_singleton_cols(denominator_broadcast, denominator)

    strata = _listify_singleton_cols(strata, denominator)
    numerator = stratify(numerator, strata+numerator_broadcast, reset_index=False)
    denominator = stratify(denominator, strata+denominator_broadcast, reset_index=False)

#     numerator = numerator.groupby(strata+index_cols+numerator_broadcast)[VALUE_COLUMN].sum()
#     denominator = denominator.groupby(strata+index_cols)[VALUE_COLUMN].sum()

    ratio = (numerator / denominator) * multiplier

    # If dropna is True, drop rows where we divided by 0
    if dropna:
        ratio.dropna(inplace=True)

    if reset_index:
        ratio.reset_index(inplace=True)

    return ratio

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
        minuend.set_index(identifier_col, append=True, inplace=True)
    else:
        subtrahend.set_index(identifier_col, append=True, inplace=True)

    # Subtract DataFrames, not Series, because Series will drop the identifier column from the index
    # if there is no broadcasting. (Behavior for Series and DataFrames is different - is this a
    # feature or a bug in pandas?)
    difference = minuend[[VALUE_COLUMN]] - subtrahend[[VALUE_COLUMN]]
    difference = difference.reset_index()

    # Add a column to specify what was subtracted from (the minuend) or what was subtracted (the subtrahend)
    colname, value = ('subtracted_from', minuend_id) if minuend_id is not None else ('subtracted_value', subtrahend_id)
    difference.insert(difference.columns.get_loc(identifier_col)+1, colname, value)

    return difference

def averted(measure: pd.DataFrame, baseline_scenario: str, scenario_col=None):
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
    # Subtract intervention from baseline
    averted = difference(measure, identifier_col=scenario_col, minuend_id=baseline_scenario)
    # Insert a column after the scenario column to record what the baseline scenario was
#     averted.insert(averted.columns.get_loc(scenario_col)+1, 'relative_to', baseline_scenario)
    return averted

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
