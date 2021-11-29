"""
Module providing functions and data structures for working with transformed Vivarium output
from the CIFF SAM model.
"""

import collections
import pandas as pd
from model_validation.vivarium_transformed_output import VivariumTransformedOutput
import model_validation.vivarium_output_processing as vop

DEFAULT_STRATA = ['year', 'sex', 'age']
ordered_ages = ['early_neonatal', 'late_neonatal', '1-5_months', '6-11_months', '12_to_23_months', '2_to_4', 'all_ages']
ordered_ages_dtype = pd.api.types.CategoricalDtype(ordered_ages, ordered=True)
ages_categorical = pd.Categorical(ordered_ages, categories=ordered_ages, ordered=True)
ordered_scenarios = ['baseline', 'wasting_treatment', 'sqlns']
ordered_wasting_states = [
    'severe_acute_malnutrition',
    'moderate_acute_malnutrition',
    'mild_child_wasting',
    'susceptible_to_child_wasting',
    'acute_malnutrition', # superstate comprising SAM and MAM
    'no_acute_malnutrition', # superstate comprising MILD and TMREL
]

class VivariumResults(VivariumTransformedOutput, collections.abc.MutableMapping):
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
    data = VivariumResults(orig_data) # Copies the original data to an object implementing MutableMapping
    data.table_names() # Displays available count_data tables for this model run
    # Access the tables via object attributes:
    data.deaths # deaths data table
    data.wasting_state_person_time # wasting state person time table
    """
    @classmethod
    def from_model_spec(cls, model_id, run_id=None):
        """Create a VivariumResults object from the model_id (e.g. 1.0, 1.1, 2.0, etc.) and optionally run_id
        (i.e. the folder name of the form 'yyyy_mm_dd_hh_mm_ss' indicating when the run was launched), using the
        `get_count_data_path` function to get the path to the count_data for the specified model.

        Example usage:
        --------------
        # This is a shortcut to creating an object equivalent to the one in the example from the class description
        data = VivariumResults.from_model_spec(2.3) # run_id can be omitted if there is only one run_id for the model
        # data = VivariumResults.from_model_spec(2.3, '2021_07_26_17_14_31') # does the same thing
        data.table_names() # Displays available count_data tables for this model run
        """
        return cls.from_directory(get_count_data_path(model_id, run_id))

    @classmethod
    def cleaned_from_model_spec(cls, model_id, run_id=None):
        """Create a VivariumResults object from the model_id (e.g. 1.0, 1.1, 2.0, etc.) and optionally run_id
        (i.e. the folder name of the form 'yyyy_mm_dd_hh_mm_ss' indicating when the run was launched), with
        the data tables reformatted using the `clean_transformed_data` function.

        Example usage:
        --------------
        data = VivariumResults.cleaned_from_model_spec(2.3) # run_id can be omitted if there is only one run_id for the model
        # data = VivariumResults.cleaned_from_model_spec(2.3, '2021_07_26_17_14_31') # does the same thing
        data.table_names() # Displays available count_data tables for this model run
        """
        return cls(clean_transformed_data(cls.from_model_spec(model_id, run_id)))

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        del self.__dict__[key]

    def append_all_causes_burden(self):
        """Append all-causes deaths, ylls, and ylds to these tables."""
        for measure in ('deaths', 'ylls', 'ylds'):
            if 'all_causes' not in self[measure]['cause'].unique():
                self[measure] = self[measure].append(get_all_causes_measure(self[measure]), ignore_index=True)

    def compute_sam_duration(self, strata=DEFAULT_STRATA):
        self.sam_duration = get_sam_duration(self, strata)

    def compute_mam_duration(self, strata=DEFAULT_STRATA):
        self.mam_duration = get_mam_duration(self, strata)

project_results_directory = '/ihme/costeffectiveness/results/vivarium_ciff_sam'

# Eventually, I want to write a script to recursively search the directories and
# populate this table automatically instead of hardcoding it. E.g. use Path.rglob to
# find all output.hdf files and/or count_data subdirectories in a specified directory:
# https://stackoverflow.com/questions/2186525/how-to-use-glob-to-find-files-recursively
# https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob
models = pd.DataFrame(
    [
        ['1.4', 'v1.4_adjusted_early_neonatal_lri_prevalence', '2021_06_14_18_37_42'],
        ['2.3', 'v2.3_wasting_birth_prevalence', '2021_07_26_17_14_31'],
        ['2.4', 'v2.4_corrected_fertility', '2021_08_03_15_08_32'],
        ['2.5', 'v2.5_stunting', '2021_08_05_16_17_12'],
        ['3.0', 'v3.0_sq_lns', '2021_08_16_17_54_19'],
        ['3.1', 'v3.1_sq_lns_stunting_stratified', '2021_08_24_10_28_32'],
        ['4.0', 'v4.0_wasting_treatment', '2021_09_20_14_45_25'],
        ['4.1', 'v4.1_wasting_treatment', '2021_09_24_16_36_30'],
        ['4.5.2', 'v4.5.2_x_factor', '2021_09_29_12_12_47'],
        ['4.5.3', 'v4.5.3_x_factor_targeted_exposure', '2021_11_02_20_09_56'],
        ['4.5.4', 'v4.5.4_x_factor_wasting_propensity', '2021_11_09_20_57_59'],
        ['4.5.5', 'v4.5.5_linear_scale-up_etc', '2021_11_22_09_07_22'],
        ['4.5.5', 'v4.5.5_linear_scale-up_etc', '2021_11_23_17_59_09'],
    ],
    columns=['model_id', 'model_name', 'run_id']
)

def get_count_data_path(model_id, run_id=None, models_df=models):
    """Returns the path to the count_data folder for the specified model number and run_id (timestamp string).
    If there is only one run_id in `models_df` corresponding to the requested model_id, then run_id
    does not need to be specified.
    """
    model_id = str(model_id) # Allow user to pass in a float if possible
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
    # Create a VivariumResults object with the same tables stored in `data`
    clean_data = VivariumResults(data)
    # Define a function to make the transition count dataframes better
    def clean_transition_df(df):
        df = split_measure_and_transition_columns(df)
        return df.join(extract_transition_states(df))
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
        clean_data['person_time'] = get_total_person_time(clean_data, 'wasting')

    if 'stunting_state_person_time' in data:
        # Rename mislabeled 'cause' column in `stunting_state_person_time`
        clean_data['stunting_state_person_time'] = (
            data['stunting_state_person_time'].rename(columns={'cause':'stunting_state'})
        )
    if 'disease_state_person_time' in data:
        # Rename poorly named 'cause' column in `disease_state_person_time` and add an actual cause column
        # Also rename 'disease' to 'cause' for consistency between table name and column names
        clean_data['cause_state_person_time'] = (
            clean_data['disease_state_person_time']
            .rename(columns={'cause':'cause_state'})
            # This is a hack that only works because all our diseases have 2 states named with the
            # convention 'cause' and 'susceptible_to_cause'. Ideally, the cause name should be
            # recorded directly by the simulation instead, unless we can guarantee that all causes
            # will have state names from which the cause name can be extracted in a uniform way.
            .assign(cause=lambda df: df['cause_state'].str.replace('susceptible_to_', ''))
        )
        del clean_data['disease_state_person_time'] # Remove redundant table after renaming

    if 'disease_transition_count' in data:
        # Rename 'disease' to 'cause' for consistency between table name and column names
        clean_data['cause_transition_count'] = clean_data['disease_transition_count']
        del clean_data['disease_transition_count']

    return clean_data

def split_measure_and_transition_columns(transition_df):
    """Separates the transition from the measure in the strings in the 'measure'
    columns in a transition count dataframe, and puts these in separate 'transition'
    and 'measure' columns.
    """
    return (transition_df
            .assign(transition=lambda df: df['measure'].str.replace('_event_count', ''))
            .assign(measure='transition_count') # Name the measure 'transition_count' rather than 'event_count'
           )

def extract_transition_states(transition_df):
    """Gets the 'from state' and 'to state' from the transitions in a transition count dataframe,
    after the transition has been put in its own 'transition' column by the `split_measure_and_transition_columns`
    function.
    """
    states_from_transition_pattern = r"^(?P<from_state>\w+)_to_(?P<to_state>\w+)$"
    # Renaming the 'susceptible_to' states is a hack to deal with the fact there's not a unique string
    # separating the 'from' and 'to' states -- it should be '__to__' instead of '_to_' or something
    states_df = (
        transition_df['transition']
        .str.replace("susceptible_to", "without") # Remove word 'to' from all states so we can split transitions on '_to_'
        .str.extract(states_from_transition_pattern) # Create dataframe with 'from_state' and 'to_state' columns
        .apply(lambda col: col.str.replace("without", "susceptible_to")) # Restore original state names
    )
    return states_df

def assert_cause_person_time_equal(data):
    """Raise an AssertionError if different cause state observers recorded different amounts of total person-time."""
    cause_person_time = get_total_person_time(data, 'cause')
    first_cause, *remaining_causes = cause_person_time.cause.unique()
    person_time = cause_person_time.query("cause==@first_cause").drop(columns='cause')
    # Check that total person-time is the same for all causes
    for cause in remaining_causes:
        vop.assert_values_equal(person_time, cause_person_time.query("cause==@cause").drop(columns='cause'))

def get_age_group_bins(*cut_points):
    """Split ages into n+1 bins beginning with the specified age groups, where n is the number
    of age groups (cut points) passed.
    """
    cut_points = [ages_categorical[0], *cut_points]
    bins = [
        ages_categorical[(ages_categorical>=start) & (ages_categorical<stop)]
        for start, stop in zip(cut_points[:-1], cut_points[1:])
    ] + [ages_categorical[ages_categorical>=cut_points[-1]]]
    return tuple(bins)

def age_to_ordered_categorical(df, inplace=False):
    if inplace:
        df['age'] = df['age'].astype(ordered_ages_dtype)
    else:
        return df.assign(age=df['age'].astype(ordered_ages_dtype))

def column_to_ordered_categorical(df, colname, ordered_categories, inplace=False):
    """Converts the column `colname` of the DataFrame `df` to an orderd pandas Categorical.
    This is useful for automatically displaying unique column elements in a specified order
    in results tables or plots.
    """
    categorical = pd.Categorical(df[colname], categories=ordered_categories, ordered=True)
    if inplace:
        df[colname] = categorical
    else:
        return df.assign(**{colname: categorical})

def to_ordered_categoricals(df, inplace=False):
    """Converts "standard" columns of df into Categoricals with their standard order."""
    colnames = ['age', 'scenario', 'wasting_state']
    orders = [ordered_ages, ordered_scenarios, ordered_wasting_states]
    for colname, order in zip(colnames, orders):
        if colname in df:
            temp = column_to_ordered_categorical(df, colname, order, inplace)
            if not inplace:
                df = temp
    if not inplace:
        return df

def get_all_ages_person_time(person_time_df, append=False):
    """Compute all-ages person time from person time stratified by age."""
    all_ages_pt = vop.marginalize(person_time_df, 'age').assign(age='all_ages')[person_time_df.columns]
    if append:
        return person_time_df.append(all_ages_pt, ignore_index=True)
    else:
        return all_ages_pt

def get_total_person_time(data, entity, include_all_ages=False):
    """Compute total person-time from person-time stratified additionally by risk state or cause state
    (i.e. "state person-time"), by marginalizing the "{entity}_state" column for the specified entity
    (one of 'wasting', 'stunting', or 'cause').
    """
    if not include_all_ages:
        # Keep all strata except entity_state
        person_time = vop.marginalize(data[f"{entity}_state_person_time"], f"{entity}_state").assign(measure='person_time')
#         person_time = vop.stratify(data[table_name], strata).assign(measure='person_time')
    else:
        # Use recursion to first get age-stratified person-time, then append all ages person-time
        person_time = get_all_ages_person_time(get_person_time(data, entity, include_all_ages=False), append=True)
    return person_time

def get_all_causes_measure(measure_df, append=False):
    """Compute all-cause deaths, ylls, or ylds (generically, measure) from cause-stratified measure."""
    all_causes_measure = vop.marginalize(measure_df, 'cause').assign(cause='all_causes')[measure_df.columns]
    if append:
        return measure_df.append(all_causes_measure, ignore_index=True)
    else:
        return all_causes_measure

def find_person_time_tables(data, colnames, exclude=None):
    """Generate person-time table names in data that contain the specified column names,
    excluding the specified table names.
    """
    colnames = set(vop.list_columns(colnames))
    exclude = vop.list_columns(exclude, default=[])
    table_names = (
        table_name for table_name, table in data.items()
        if table_name not in exclude and table_name.endswith("person_time")
        and colnames.issubset(table.columns)
    )
    return table_names

def get_person_time_table_name(data, colnames, exclude=None):
    """Return the name of a person-table that contains the specified columns,
    or raise a ValueError if none can be found.
    """
    try:
        person_time_table_name = next(find_person_time_tables(data, colnames, exclude))
    except StopIteration:
        raise ValueError(f"No person-time table found with columns {colnames}."
                         f" (Excluded tables: {exclude})")
    return person_time_table_name

def get_prevalence(data, state_variable, strata, prefilter_query=None, **kwargs):
    """Compute the prevalence of the specified state_variable, which may represent a risk state or cause state
    (one of 'wasting_state', 'stunting_state', or 'cause_state'), or another stratification variable
    tracked in the simulation (e.g. 'sq_lns', 'wasting_treatment', or 'x_factor').
    `prefilter_query` is a query string passed to the DataFrame.query() function of both the
    numerator and denominator before taking the ratio. This is useful for aggregating over strata
    when computing the prevalence of a subset of the population.
    The `kwargs` dictionary stores keyword arguments to pass to the vivarium_output_processing.ratio()
    function.
    """
    # Broadcast the numerator over the state variable to compute the prevalence of each state
    kwargs['numerator_broadcast'] = vop.list_columns(
        state_variable, kwargs.get('numerator_broadcast'), default=[])
    # Determine columns we need for numerator and denominator so we can look up appropriate person-time tables
    numerator_columns = vop.list_columns(strata, kwargs['numerator_broadcast'])
    denominator_columns = vop.list_columns(strata, kwargs.get('denominator_broadcast'), default=[])
    # Define numerator
    if f"{state_variable}_person_time" in data:
        state_person_time = data[f"{state_variable}_person_time"]
    else:
        # Find a person-time table that contains necessary columns for numerator.
        # Exclude cause-state person-time because it contains total person-time multiple times,
        # which would make us over-count.
        numerator_table_name = get_person_time_table_name(data, numerator_columns, exclude='cause_state_person_time')
        state_person_time = data[numerator_table_name]
    # Find a person-time table that contains necessary columns for total person-time in the denominator.
    # Exclude cause-state person-time because it contains total person-time multiple times,
    # which would make us over-count.
    denominator_table_name = get_person_time_table_name(data, denominator_columns, exclude='cause_state_person_time')
    person_time = data[denominator_table_name]
    # Filter input dataframes if requested
    if prefilter_query is not None:
        state_person_time = state_person_time.query(prefilter_query)
        person_time = person_time.query(prefilter_query)
    # Divide to compute prevalence
    prevalence = vop.ratio(
        numerator=state_person_time,
        denominator=person_time,
        strata=strata,
        **kwargs, # Includes numerator_broadcast over state_variable
    ).assign(measure='prevalence')
    return prevalence

def get_transition_rates(data, entity, strata, prefilter_query=None, **kwargs):
    """Compute the transition rates for the given entity (either 'wasting' or 'cause')."""
    # We need to match transition count with person-time in its from_state. We do this by
    # renaming the entity_state column in state_person_time df, and adding from_state to strata.
    transition_count = data[f"{entity}_transition_count"]
    state_person_time = data[f"{entity}_state_person_time"].rename(columns={f"{entity}_state": "from_state"})
    strata = vop.list_columns(strata, "from_state")

    # Filter the numerator and denominator if requested
    if prefilter_query is not None:
        transition_count = transition_count.query(prefilter_query)
        state_person_time = state_person_time.query(prefilter_query)

    # Broadcast numerator over transition (and redundantly, to_state) to get the transition rate across
    # each arrow separately. Without this broadcast, we'd get the sum of all rates out of each state.
    kwargs['numerator_broadcast'] = vop.list_columns(
        'transition', 'to_state', kwargs.get('numerator_broadcast'), df=transition_count, default=[])
    # Divide to compute the transition rates
    transition_rates = vop.ratio(
        transition_count,
        state_person_time,
        strata = strata,
        **kwargs
    ).assign(measure='transition_rate')
    return transition_rates

def get_relative_risk(data, measure, outcome, strata, factor, reference_category, prefilter_query=None):
    """
    `measure` is one of 'prevalence', 'transition_rate', or 'mortality_rate'.
        Each of these has a different type of table for the numerator (person time, transition_count, or deaths).
    `outcome` is passed to either get_transition_rates or get_prevalence,
        and represents the outcome for which we want to compute the relative risk (e.g. 'stunting_state',
        'wasting_state', or a stratification variable for measure=='prevalence', or 'wasting' or 'cause' for
        measure=='transition_rate', or 'cause'??? or 'death'??? or None??? or cause_name???
        for measure=='mortality_rate').
        Note that `outcome` may be sort of a "meta-description" of the outcome we're interested in,
        with the actual outcome being one or more items described by this variable (e.g. the specific
        stunting or wasting categories, specific wasting state or cause state transitions, or deaths from
        a specific cause).
    `factor` is the risk factor or other stratifying variable for which we want to compute the relative risk
        (e.g. x_factor, sq_lns, stunting_state, wasting_state).
    `reference_category` is the factor category to put in the denominator to use as a reference for computing
        relative risks (e.g. the TMREL). The numerator will be broadcast over all remaining categories.
    """
    if measure=='prevalence':
        get_measure = get_prevalence
        ratio_strata = vop.list_columns(strata, outcome)
    elif measure=='transition_rate':
        get_measure = get_transition_rates
        ratio_strata = vop.list_columns(strata, 'transition', 'from_state', 'to_state')
    elif measure=='mortality_rate': # Or burden_rate, and then pass 'death', 'yll', or 'yld' for outcome
#         get_measure = get_rates # or get_burden_rates
#         ratio_strata = vop.list_columns(strata, ???)
        raise NotImplementedError("relative mortality rates have not yet been implemented")
    else:
        raise ValueError(f"Unknown measure: {measure}")
    # Add risk factor to strata in order to get prevalence or rate in different risk factor categories
    measure_df = get_measure(data, outcome, vop.list_columns(strata, factor), prefilter_query)
    numerator = (measure_df.query(f"{factor} != '{reference_category}'")
                 .rename(columns={f"{factor}":f"numerator_{factor}"}))
    denominator = (measure_df.query(f"{factor} == '{reference_category}'")
                   .rename(columns={f"{factor}":f"denominator_{factor}"}))
    relative_risk = vop.ratio(
        numerator,
        denominator,
        ratio_strata, # Match outcome categories to compute the relative risk
        numerator_broadcast=f"numerator_{factor}",
        denominator_broadcast=f"denominator_{factor}",
    ).assign(measure='relative_risk') # Or perhaps I should be more specific, i.e. "prevalence_ratio" or "rate_ratio"
    return relative_risk

def get_sam_duration(data, strata):
    sam_person_time = data.wasting_state_person_time.query(
        "wasting_state == 'severe_acute_malnutrition'")
    transitions_into_sam = data.wasting_transition_count.query(
        "transition == 'moderate_acute_malnutrition_to_severe_acute_malnutrition'")
    sam_duration = vop.ratio(
        sam_person_time,
        transitions_into_sam,
        strata=strata
    )
    return sam_duration

def get_mam_duration(data, strata):
    mam_person_time = data.wasting_state_person_time.query(
        "wasting_state == 'moderate_acute_malnutrition'")
    mild_to_mam = data.wasting_transition_count.query(
        "transition == 'mild_child_wasting_to_moderate_acute_malnutrition'")
    sam_to_mam = data.wasting_transition_count.query(
        "transition == 'severe_acute_malnutrition_to_moderate_acute_malnutrition'")
    transitions_into_mam = vop.value(mild_to_mam, exclude='transition') + vop.value(sam_to_mam, exclude='transition')
    mam_duration = vop.ratio(
        mam_person_time,
        transitions_into_mam,
        strata=strata
    )
    return mam_duration

def get_x_factor_wasting_transition_rate_ratio(data:VivariumResults, strata):
    """Computes the ratios of incidence rates into Mild, MAM, and SAM for simulants with
    X-factor to the incidence rates for simulants without X-factor.
    """
    under_6mo, over_6mo, all_ages = map(list, get_age_group_bins('6-11_months', 'all_ages'))
    # Wasting state transition rates
    prefilter_query=f"age in {over_6mo}"
    # Need to stratify by X-factor to get transition rates with/without X-factor
    transition_rates = get_transition_rates(
        data, 'wasting', vop.list_columns(strata, 'x_factor'), prefilter_query)

    # Reference category = without X-factor (denominator)
    transition_rate_without_x_factor =(
        transition_rates.query("x_factor=='cat2'")
        .rename(columns={'x_factor':'denominator_x_factor'})
    )
    # Non-reference category(ies) = with X-factor (numerator)
    transition_rate_with_x_factor =(
        transition_rates.query("x_factor!='cat2'")
        .rename(columns={'x_factor':'numerator_x_factor'})
    )
    # Compute transition rate ratio
    transition_rate_ratio = vop.ratio(
        transition_rate_with_x_factor,
        transition_rate_without_x_factor,
        strata=vop.list_columns(strata, 'transition', 'from_state', 'to_state'),
        numerator_broadcast='numerator_x_factor',
        denominator_broadcast='denominator_x_factor',
    ).assign(measure='rate_ratio')
    return transition_rate_ratio

def get_sqlns_mam_incidence_ratio(data:VivariumResults):
    """Computes the incidence rate ratio of MAM for SQLNS-covered vs. SQLNS-uncovered.
    The computed incidence rate ratios are stratified by age, sex, and year.
    """
    # Get a list of age groups over 6 months (there is no sqlns treatment under 6 months)
    over_6mo = list(ages_categorical[(ages_categorical >= '6-11_months') & (ages_categorical != 'all_ages')])
    # Get a query string to filter to rows with nonzero coverage
    nonzero_coverage_query = "scenario not in ['baseline', 'wasting_treatment'] and age in @over_6mo and year > '2022'"
    # Filter wasting transitions dataframe to mild->mam transition and strata with nonzero sqlns coverage
    mild_to_mam_count = data.wasting_transition_count.query(
        f"transition =='mild_child_wasting_to_moderate_acute_malnutrition' and ({nonzero_coverage_query})"
    )
    strata = ['year', 'sex', 'age']
    # Get person-time in MILD wasting for strata with nonzero sqlns coverage
    mild_wasting_person_time = data.wasting_state_person_time.query(
        f"wasting_state=='mild_child_wasting' and ({nonzero_coverage_query})"
    )
    mam_incidence_rate_by_coverage = vop.ratio(
        mild_to_mam_count,
        mild_wasting_person_time,
        strata = strata + ['sq_lns'],
    )
    assert mam_incidence_rate_by_coverage.value.notna().all(), "unexpected NaNs!"
    mam_incidence_rate_covered = (
        mam_incidence_rate_by_coverage.query("sq_lns=='covered'")
        .assign(measure="mam_incidence_rate_among_sqlns_covered")
    )
    mam_incidence_rate_uncovered = (
        mam_incidence_rate_by_coverage.query("sq_lns=='uncovered'")
        .assign(measure="mam_incidence_rate_among_sqlns_uncovered")
    )
    assert mam_incidence_rate_covered.shape == mam_incidence_rate_uncovered.shape, "unmatched strata!"
    mam_incidence_rate_ratio = vop.ratio(
        mam_incidence_rate_covered,
        mam_incidence_rate_uncovered,
        strata=strata,
    )
    return mam_incidence_rate_ratio

def get_sqlns_risk_prevalence_ratio(data:VivariumResults, risk_name:str, stratify_by_year:bool):
    """Computes the prevalence ratio of each stunting or wasting category for SQ-LNS-covered vs. SQ-LNS-uncovered.
    The prevalence ratio is for verifying the correct effect size of SQ-LNS on stunting or wasting.

    If `stratify_by_year` is True, the risk category prevalences will be computed separately for each year, whereas
    if `stratify_by_year` is False, all years in the simulation will be pooled to compute the prevalence.

    Note that we must stratify by age and sex since stunting and wasting prevalence varies across these demographic
    strata, but we are using the same prevalence for all years, so we'll get more data per stratum
    if we omit year stratification.
    """
    # Get the table name for stunting or wasting
    risk_person_time_table = f"{risk_name}_state_person_time"
    # Get a list of age groups under 6 months (there is no sqlns treatment in these age groups)
    under_6mo = list(ages_categorical[ages_categorical < '6-11_months'])
    # Set the demographic strata based on whether we're stratifying by year
    demographic_strata = ['year', 'sex', 'age'] if stratify_by_year else ['sex', 'age']
    # Get total person-time in each stratum defined by demographics and sqlns coverage
    person_time_by_sqlns_coverage = get_person_time(
        data, demographic_strata+['sq_lns'], risk_person_time_table, include_all_ages=False
    )
    # Compute risk prevalence in sqlns-covered group and in sqlns-uncovered group
    risk_prevalence_by_coverage = vop.ratio(
        data[risk_person_time_table],
        person_time_by_sqlns_coverage,
        strata=demographic_strata+['sq_lns'], # stratify by sqlns coverage
        numerator_broadcast=f'{risk_name}_state', # compute the prevalence of each risk category
    )
    # Check that NaN's occur precisely where we expect coverage to be 0 (which results in division by zero)
    zero_coverage_query = "scenario in ['baseline', 'wasting_treatment'] or age in @under_6mo"
    if stratify_by_year:
        zero_coverage_query += " or year == '2022'" # treatment doesn't start until 2023
    assert risk_prevalence_by_coverage.query(
        f"sq_lns == 'covered' and ({zero_coverage_query})"
    ).equals(risk_prevalence_by_coverage.query('value!=value')),\
    f"Unexpected NaNs in {risk_name} prevalence by sqlns coverage!" # Note: value!=value iff value==NaN

    # Filter to strata where there is nonzero coverage once we're sure there's no problem
    risk_prevalence_by_coverage = risk_prevalence_by_coverage.query(f"~({zero_coverage_query})")
    # Get separate dataframes for sqlns-covered vs. sqlns-uncovered
    risk_prevalence_covered = (
        risk_prevalence_by_coverage.query("sq_lns == 'covered'")
        .assign(measure='prevalence_among_sqlns_covered')
    )
    risk_prevalence_uncovered = (
        risk_prevalence_by_coverage.query("sq_lns == 'uncovered'")
        .assign(measure='prevalence_among_sqlns_uncovered')
    )
    # We should have exactly the same strata in covered and uncovered, so the shapes should be equal
    assert risk_prevalence_covered.shape == risk_prevalence_uncovered.shape,\
    f"Unmatched strata for {risk_name} prevalence in sqlns-covered vs. sqlns-uncovered!"
    # Compute the risk prevalence ratio of sqlns-covered to sqlns-uncovered for each risk category
    risk_prevalence_ratio = vop.ratio(
        risk_prevalence_covered,
        risk_prevalence_uncovered,
        strata=demographic_strata+[f'{risk_name}_state'], # match risk categories to compute the ratio
    )
    return risk_prevalence_ratio
