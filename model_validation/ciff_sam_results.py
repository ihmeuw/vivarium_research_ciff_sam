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
        del self.key

    def compute_total_person_time(self, include_all_ages=True):
        """Compute and store total person-time from wasting-state person-time."""
        self.person_time = get_person_time(self, DEFAULT_STRATA, 'wasting_state_person_time', include_all_ages)

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
    # Create a VivariumResults object with the same tables stored in `data`
    clean_data = VivariumResults(data)
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
    if 'stunting_state_person_time' in data:
        # Rename mislabeled 'cause' column in `stunting_state_person_time`
        clean_data['stunting_state_person_time'] = (
            data['stunting_state_person_time'].rename(columns={'cause':'stunting_state'})
        )
    if 'disease_state_person_time' in data:
        # Rename poorly named 'cause' column in `disease_state_person_time` and add an actual cause column
        clean_data['disease_state_person_time'] = (
            data['disease_state_person_time']
            .rename(columns={'cause':'cause_state'})
            .assign(cause=lambda df: df['cause_state'].str.replace('susceptible_to_', ''))
        )
    return clean_data

def age_to_ordered_categorical(df, inplace=False):
    if inplace:
        df['age'] = df['age'].astype(ordered_ages_dtype)
    else:
        return df.assign(age=df['age'].astype(ordered_ages_dtype))

def get_all_ages_person_time(person_time):
    """Compute all-ages person time from person time stratified by age."""
    return vop.marginalize(person_time, 'age').assign(age='all_ages')[person_time.columns]

def get_person_time(data, strata, table_name, include_all_ages=False):
    """Compute total person-time stratified by strata, from person-time stratified additionally
    by risk state or disease state.
    """
    if not include_all_ages:
#         person_time = vop.marginalize(data.wasting_state_person_time, 'wasting_state').assign(measure='person_time')
        person_time = vop.stratify(data[table_name], strata).assign(measure='person_time')
    else:
        person_time = get_person_time(data, strata, table_name, False)
        person_time = person_time.append(get_all_ages_person_time(person_time), ignore_index=True)
    return person_time

def get_all_causes_measure(measure):
    """Compute all-cause deaths, ylls, or ylds (generically, measure) from cause-stratified measure."""
    return vop.marginalize(measure, 'cause').assign(cause='all_causes')[measure.columns]

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

def get_sqlns_coverage(data, strata):
    sqlns_coverage = vop.ratio(
        data.wasting_state_person_time,
        data.person_time.query("age != 'all_ages'"),
        strata=strata,
        numerator_broadcast='sq_lns'
    )
    return sqlns_coverage

def get_sqlns_mam_incidence_ratio(data:VivariumResults):
    # Get a list of age groups over 6 months (there is no sqlns treatment in these age groups)
    over_6mo = list(ages_categorical[(ages_categorical >= '6-11_months') & (ages_categorical != 'all_ages')])
    # Get a query string to filter to rows with nonzero coverage
    nonzero_coverage_query = "scenario=='treatment_and_prevention' and age in @over_6mo and year > '2022'"
    # Filter wasting transitions dataframe to mild->mam transition and strata with nonzero sqlns coverage
    mild_to_mam_count = data.wasting_transition_count.query(
        f"transition =='mild_child_wasting_to_moderate_acute_malnutrition' and ({nonzero_coverage_query})"
    )
    strata = ['year', 'sex', 'age']
#     person_time_by_sqlns_coverage = get_person_time_by_sqlns_coverage(data, 'wasting', strata)
#     person_time_by_sqlns_coverage = get_person_time(
#         data, vop.list_columns(strata, 'sq_lns'), 'wasting_state_person_time', include_all_ages=False
#     ).query(nonzero_coverage_query)
    mild_wasting_person_time = data.wasting_state_person_time.query(
        f"wasting_state=='mild_child_wasting' and ({nonzero_coverage_query})"
    )
    mam_incidence_rate_by_coverage = vop.ratio(
        mild_to_mam_count,
        mild_wasting_person_time,
        strata=vop.list_columns(strata, 'sq_lns', df=mild_wasting_person_time),
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
    zero_coverage_query = "scenario == 'baseline' or age in @under_6mo"
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
