# From 2021_08_24a_ciff_sam_v2.5_check_stuff.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 8)

import collections

import warnings
# warnings.filterwarnings('ignore')
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import Path

import db_queries as db
import vivarium_helpers.id_helper as idh

# Add the repo directory vivarium_research_ciff_sam/ to sys.path
import os, sys
repo_path = os.path.abspath('../..')
sys.path.append(repo_path)
# Assumes vivarium_research_ciff_sam/ is in sys.path
import model_validation.vivarium_transformed_output as vto
# import model_validation.vivarium_raw_output as vro
import model_validation.vivarium_output_processing as vo

# From 2021_07_30a_vivarium_data_scratch_work.ipynb

class CountData2(collections.abc.MutableMapping):
    def __init__(self, table_dict):
        self.__dict__ = dict(table_dict)
#         for table_name, table in table_dict.items():
#             self[table_name] = table
    
#     def __setattr(self, key, value):
#         if not isinstance(value, pd.DataFrame):
#             raise TypeError(f"Only DataFrames are allowed in CountData. You provided {type(value)}")
#         super().__setattr__(key, value)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def __delitem__(self, key):
        del self.key
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)
    
    def to_dict(self):
        # return dict(self) # this also works since we implement Mapping
        return dict(self.__dict__)
    
    def table_names(self):
        return list(self.keys())

# From 2021_08_03a_vivarium_output_scratch_work.ipynb

def make_variable_name(string): return re.sub('\W+|^(?=\d)','_', string)

# From 2021_08_11_pull_lbwsg_data.ipynb

def get_draw_cols(draws_or_df=None):
    if isinstance(draws_or_df, pd.DataFrame):
        draw_cols = draws_or_df.columns.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    else:
        if draws_or_df is None:
            draws_or_df = range(1000) # draws
        draw_cols = [f'draw_{i}' for i in draws_or_df]
    return draw_cols

def get_draw_numbers(draw_cols_or_df=None):
    if isinstance(draw_cols_or_df, pd.DataFrame):
        draw_cols_or_df = draw_cols_or_df.columns.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    if draw_cols_or_df is not None:
        draws = [int(draw.replace('draw_', '')) for draw in draw_cols_or_df]
    else:
        draws = list(range(1000))
    return draws

# From 2021_08_24a_ciff_sam_v2.5_check_stuff.ipynb

project_results_dir = '/ihme/costeffectiveness/results/vivarium_ciff_sam'
model_name = 'v2.5_stunting'
model_timestamp = '2021_08_05_16_17_12'
username = 'ndbs'

model_count_data_dir = f'{project_results_dir}/{model_name}/ciff_sam/{model_timestamp}/count_data/'

project_vv_directory_name = 'ciff_malnutrition/verification_and_validation'

output_dir = f'/ihme/homes/{username}/vivarium_results/{project_vv_directory_name}/{model_name}'
share_output_dir = f'/share/scratch/users/ndbs/vivarium_results/{project_vv_directory_name}/{model_name}'
j_output_dir = f'/home/j/Project/simulation_science/{project_vv_directory_name}/{model_name}'

# Create the output directories if they don't exist
# Note from Path.mkdir() documentation:
#   "If mode is given, it is combined with the process’ umask value to determine the file mode and access flags."
#
# I don't know what this notebook process' umask value will be, so I don't know if this will actually result
# in the correct (most permissive) permissions for the directories...
for directory in [output_dir, share_output_dir, j_output_dir]:
    Path(directory).mkdir(mode=0o777, parents=True, exist_ok=True)

def _ensure_columns_not_levels(df, column_list=None):
    """Move Index levels into columns to enable passing index level names as well as column names."""
    if column_list is None: column_list = []
    if df.index.nlevels > 1 or df.index.name in column_list:
        df = df.reset_index()
    return df

def describe(data, **describe_kwargs):
    if 'percentiles' not in describe_kwargs:
        describe_kwargs['percentiles'] = [.025, .975]
    data = _ensure_columns_not_levels(data, [vo.DRAW_COLUMN, vo.VALUE_COLUMN])
    groupby_cols = data.columns.difference([vo.DRAW_COLUMN, vo.VALUE_COLUMN]).to_list()
    return data.groupby(groupby_cols)[vo.VALUE_COLUMN].describe(**describe_kwargs)

def get_all_ages_person_time(person_time):
    return vo.marginalize(person_time, 'age').assign(age='all')[person_time.columns]

def get_total_person_time(data, include_all_ages=False):
    if not include_all_ages:
        person_time = vo.marginalize(data.wasting_state_person_time, 'wasting_state').assign(measure='person_time')
    else:
        person_time = get_total_person_time(data, False)
        person_time = person_time.append(get_all_ages_person_time(person_time), ignore_index=True)
    return person_time

def clean_transformed_data(data):
    """Reformat transformed count data to make more sense."""
    # Rename mislabeled 'cause' column in `wasting_state_person_time`
    wasting_state_person_time = data.wasting_state_person_time.rename(columns={'cause':'wasting_state'})
    # Rename poorly named 'cause' column in `disease_state_person_time` and add an actual cause column
    disease_state_person_time = (
        data.disease_state_person_time
        .rename(columns={'cause':'cause_state'})
        .assign(cause=lambda df: df['cause_state'].str.replace('susceptible_to_', ''))
    )
    # Define a function to make the transition count dataframes better
    def clean_transition_df(df):
        return (df
                .assign(transition=lambda df: df['measure'].str.replace('_event_count', ''))
                .assign(measure='transition_count')
               )
    # Make the wasting and disease transition count dataframes better
    wasting_transition_count, disease_transition_count = map(
        clean_transition_df, (data.wasting_transition_count, data.disease_transition_count)
    )
    # Create a dictionary with the original or cleaned dataframes and create a cleaned Output object
    data_dict = data.to_dict()
    data_dict.update(
        {'wasting_state_person_time': wasting_state_person_time,
         'disease_state_person_time': disease_state_person_time,
         'wasting_transition_count': wasting_transition_count,
         'disease_transition_count': disease_transition_count,
        }
    )
    clean_data = vto.VivariumTransformedOutput(data_dict)
    return clean_data

# From https://github.com/ihmeuw/vivarium_research_lsff/blob/main/gbd_data_summary/pull_gbd2019_data.py#L623
def aggregate_mean_lower_upper(df_or_groupby):
    """"""
    def lower(x): return x.quantile(0.025)
    def upper(x): return x.quantile(0.975)
    return df_or_groupby.agg(['mean', lower, upper])

# From 2021_09_03b_ciff_sam_v3.1_scratch__gitignore__.ipynb
# check_allclose(pt_from_stunting, data.person_time)
# pd.testing.assert_frame_equal(pt_from_stunting, data.person_time)
def check_allclose(df1, df2):
    df1 = vop.value(df1)
    df2 = vop.value(df2).reindex(df1.index)
    return np.allclose(df1.value, df2.value)

def assert_values_equal(df1, df2):
    df1 = vop.value(df1)
    df2 = vop.value(df2).reindex(df1.index)
    pd.testing.assert_frame_equal(df1, df2)

# From 2021_09_09a_ciff_sam_v3.1_vv.ipynb
def normal_dist_from_mean_lower_upper(mean, lower, upper, quantile_ranks=(0.025,0.975)):
    """Returns a frozen normal distribution with the specified mean, such that
    (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]).
    """
    # quantiles of the standard normal distribution with specified quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantile_ranks)
    stdev = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    # Frozen normal distribution
    return stats.norm(loc=mean, scale=stdev)

def plot_pdf(dist, label, ax=None):
    if ax is None: ax = plt.gca()
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    ax.plot(x, dist.pdf(x), lw=2, alpha=0.8, label=label)

# From 2021_10_05a_ciff_sam_v4.5_x_factor_vv_check_wasting_prevalence.ipynb
def get_age_group_bins(*cut_points):
    cut_points = [csr.ages_categorical[0], *cut_points]
#     print(cut_points, list(zip(cut_points[:-1], cut_points[1:])))
    bins = [
#         print(start, stop)
        csr.ages_categorical[(csr.ages_categorical>=start) & (csr.ages_categorical<stop)]
        for start, stop in zip(cut_points[:-1], cut_points[1:])
    ] + [csr.ages_categorical[csr.ages_categorical>=cut_points[-1]]]
    return tuple(bins)

print(get_age_group_bins(), '\n')
print(get_age_group_bins('early_neonatal'), '\n')
print(get_age_group_bins('12_to_23_months'), '\n')
print(get_age_group_bins('6-11_months', 'all_ages'))

model_path = '/share/costeffectiveness/results/vivarium_ciff_sam/v4.5.2_x_factor/ciff_sam/2021_09_29_12_12_47'
with open(f"{model_path}/keyspace.yaml", 'r') as file:
    keyspace = yaml.load(file, Loader=yaml.SafeLoader)

keyspace
pd.json_normalize(keyspace).T

# From 2021_10_25a_v4.1_calculate_percent_wasting_reduction_for_10_26_call.ipynb
def aggregate_wasting_states(df, append=False):
    """Aggrgates (by summing) the SAM and MAM wasting states into an 'acute_malnutrition' "superstate",
    and aggregates (by summing) the TMREL and MILD wasting states into a 'no_acute_malnutrition' "superstate".
    Returns a new dataframe with the same columns as the argument df.
    If append is False (default), the new dataframe contains only the aggregated superstates,
    in the 'wasting_state' column.
    If append is True, the aggregated superstates are appended to the end of the dataframe df, and this
    new concatnated dataframe is returned.

    Note: This function could be generalized to aggregate specified categories in any specified column into
    "supercategories" for that column. I have come across at least three other situations where this would
    be useful: aggregating age groups into 'all_ages' or, e.g., 'over_6_months'; aggregating causes into
    'all_causes'; and aggregating transition counts to calculate the total inflow into a wasting state.
    """
    superstate_to_states = {
        'acute_malnutrition': ['moderate_acute_malnutrition', 'severe_acute_malnutrition'],
        'no_acute_malnutrition': ['susceptible_to_child_wasting', 'mild_child_wasting'],
    }
    state_to_superstate = {
        state: superstate for superstate, states in superstate_to_states.items() for state in states
    }
    aggregated_df = (
        df.rename(columns={'wasting_state':'orig_wasting_state'})
        .assign(wasting_state=lambda df: df['orig_wasting_state'].map(state_to_superstate))
        .pipe(vp.marginalize, 'orig_wasting_state')
    )
    if append:
        return df.append(aggregated_df, ignore_index=True)
    else:
        return aggregated_df

def use_custom_index(func, index_cols):
    """Function wrappeer returning a wrapped function that temporarily resets the INDEX_COLUMNS
    global variable in the vivarium_output_processing module to the specified `index_cols` argument,
    then calls the passed function `func`, then resets INDEX_COLUMNS to their original values.
    """
    def custom_index_wrapped_function(*args, **kwargs):
        orig_index_cols = vp.INDEX_COLUMNS
        vp.set_global_index_columns(vp.list_columns(index_cols))
        try:
            return func(*args, **kwargs)
        finally:
            # Make sure we reset the global index columns to their original values even if an error is raised
            vp.set_global_index_columns(orig_index_cols)
    return custom_index_wrapped_function

def relative_reduction(measure_df, strata, baseline_scenario, scenario_col='scenario', **kwargs):
    """Compute the relative reduction of a measure from baseline scenario to each intervention scenario.
    That is, computes ((baseline value) - (intervention value)) / (baseline value) for each intrvention
    scenario.
    """
    reduction_from_baseline = vp.averted(measure_df, baseline_scenario, scenario_col)
    baseline_measure = (
        measure_df.query(f"{scenario_col}=={baseline_scenario!r}")
        .rename(columns={'scenario': 'reference_scenario'}))
    kwargs['numerator_broadcast'] = vp.list_columns(
        'scenario', kwargs.get('numerator_broadcast'), default=[])
    kwargs['denominator_broadcast'] = vp.list_columns(
        'reference_scenario', kwargs.get('denominator_broadcast'), default=[])
    relative_reduction = use_custom_index(vp.ratio, vp.DRAW_COLUMN)(
        reduction_from_baseline,
        baseline_measure,
        strata=strata,
        **kwargs
    )
    return relative_reduction

# From 2021_11_05b_v4.5.3_vv_x_factor_prevalence_and_load_model_spec_yaml.ipynb
count_data_path = csr.get_count_data_path('4.5.3')
output_path = count_data_path[:-len('count_data/')]

model_spec_path = f"{output_path}/model_specification.yaml"
with open(model_spec_path, 'r') as model_spec_file:
    model_spec = yaml.load(model_spec_file, Loader=yaml.SafeLoader)

def tabify(s):
    subkeys = s.split('.')
    tabs = ':\n'
    result = subkeys[0]
    for key in subkeys[1:]:
        tabs+='\t'
        result += tabs+key
    return result

tabified = tabify('configuration.metrics.child_wasting_observer.by_age')
print(repr(tabified))
print(tabified)
print(tabified.expandtabs(4))

def print_model_spec(model_spec):
    df = pd.json_normalize(model_spec)
#     keys = df.columns.str.replace('.', ':\n', regex=False)
    keys = df.columns.map(tabify)
    lines = keys + ": " + df.loc[0].astype(str)
    print("\n".join(lines)) 
print_model_spec(model_spec)

with open(model_spec_path, 'r') as model_spec_file:
    model_spec_file_lines = model_spec_file.readlines()
for line in model_spec_file_lines:
    print(line, end='')

with open(model_spec_path, 'r') as model_spec_file:
    model_spec_contents = model_spec_file.read()
print(model_spec_contents)

print(yaml.dump(model_spec))

model_spec

def convert_to_variable_name(string):
    """Converts a string to a valid Python variable.
    Runs of non-word characters (regex matchs \W+) are converted to '_', and '_' is appended to the
    beginning of the string if the string starts with a digit (regex matches ^(?=\d)).
    Solution copied from here:
    https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    return re.sub('\W+|^(?=\d)', '_', string)

#### From 2022_01_05a_v5.1.2_check_lbwsg_outputs ####

def get_low_birthweight_prevalence(data, strata, prefilter_query=None, **kwargs):
    if prefilter_query is None:
        prefilter_query = ''
    else:
        prefilter_query += ' and '
    numerator = data['births'].query(f"{prefilter_query}measure=='low_weight_births'")
    denominator = data['births'].query(f"{prefilter_query}measure=='total_births'")
    prevalence = vp.ratio(
        numerator,
        denominator,
        strata,
        **kwargs,
    ).assign(measure='prevalence')
    return prevalence

# Generalized version, which may or may not apply in other situations:
def get_population_prevalence(data, category, population_measure, strata, prefilter_query=None, **kwargs):
    """
    Usage:
    get_population_prevalence(data, 'low_weight', 'births', strata, prefilter_query, **kwargs)
    
    I'm calling this "population" prevalence as opposed to "person-time-weighted" prevalence,
    which is how we usually compute prevalence (e.g. in my `get_prevalence` function),
    because we're using counting measure on the population rather than person-time measure.
    """
    if prefilter_query is None:
        prefilter_query = ''
    else:
        prefilter_query += ' and '
    numerator = data[population_measure].query(f"{prefilter_query}measure=='{category}_{population_measure}'")
    denominator = data[population_measure].query(f"{prefilter_query}measure=='total_{population_measure}'")
    prevalence = vp.ratio(
        numerator,
        denominator,
        strata,
        **kwargs,
    ).assign(measure=f"prevalence")
    return prevalence

# This version doesn't quite work with my current ratio function
# because of differnt values in 'measure' column:
def get_population_prevalence2(data, population_measure, strata, prefilter_query=None, **kwargs):
    """
    Usage:
    get_population_prevalence(data, 'low_weight', 'births', strata, prefilter_query, **kwargs)
    """
    if prefilter_query is None:
        prefilter_query = ''
    else:
        prefilter_query += ' and '
    # Broadcast the numerator over the measure column to compute the prevalence of each category
    kwargs['numerator_broadcast'] = vp.list_columns(
        'measure', kwargs.get('numerator_broadcast'), default=[])
    denominator_measure=f'total_{population_measure}'
    numerator = data[population_measure].query(f"{prefilter_query}measure!={denominator_measure!r}")
    denominator = data[population_measure].query(f"{prefilter_query}measure=={denominator_measure!r}")
    prevalence = vp.ratio(
        numerator,
        denominator,
        strata,
        **kwargs,
    ).assign(measure=f"prevalence")
    return prevalence

def get_birthweight_mean(data, strata, prefilter_query=None, **kwargs):
    if prefilter_query is None:
        prefilter_query = ''
    else:
        prefilter_query += ' and '
    numerator = data['births'].query(f"{prefilter_query}measure=='birth_weight_sum'")
    denominator = data['births'].query(f"{prefilter_query}measure=='total_births'")
    mean = vp.ratio(
        numerator,
        denominator,
        strata,
        **kwargs,
    ).assign(measure='mean')
    return mean

# Generalized version, which may or may not apply in other situations:
def get_population_mean(data, attribute, population_measure, strata, prefilter_query=None, **kwargs):
    """
    Usage:
    get_population_mean(data, 'birth_weight', 'births', strata, prefilter_query, **kwargs)
    
    I'm calling this a "population" mean as opposed to a "person-time weighted" mean,
    because we're using counting measure on the population rather than person-time measure.
    """
    if prefilter_query is None:
        prefilter_query = ''
    else:
        prefilter_query += ' and '
    numerator = data[population_measure].query(f"{prefilter_query}measure=='{attribute}_sum'")
    denominator = data[population_measure].query(f"{prefilter_query}measure=='total_{population_measure}'")
    prevalence = vp.ratio(
        numerator,
        denominator,
        strata,
        **kwargs,
    ).assign(measure=f"mean")
    return prevalence

# From 2022_01_05b_check_lbwsg_exposure_from_artifact
def calculate_mean_birthweight(lbwsg_exposure, cat_df):
    """Calculates the mean birthweight according to the exposure distribution,
    assuming a uniform birthweight distribution within each LBWSG category.
    `lbwsg_exposure` is LBWSG exposure data from the Artifact
    `cat_df` is the LBWSG category data DataFrame created by Nathaniel's functions
    """
    lbwsg_exposure = (
        lbwsg_exposure
        .rename_axis(index={'parameter':'lbwsg_category'}) # rename to match cat_df index
        .rename_axis(columns='draw')
        .stack('draw')
    )
    mean_birthweight_by_cat = (
        cat_df
        .set_index('lbwsg_category')
        ['bw_midpoint'] # mean is midpoint since we're assuming uniform distribution on each category
    )
    # get groupby columns to sum over categories
    sum_index_cols = lbwsg_exposure.index.names.difference(['lbwsg_category'])
    mean_birthweight = (
        (lbwsg_exposure * mean_birthweight_by_cat) # prevalence-weighted mean birthweight by category
        .groupby(sum_index_cols)
        .sum()
        .rename('mean_birthweight')
        .unstack('draw') # put back into format from artifact
    )
    return mean_birthweight

def calculate_low_birthweight_prevalence(lbwsg_exposure, cat_df, low_bw_cutoff=2500):
    """Calculates prevalence of births with birthweight <= 2500g or some other cutoff.
    The cutoff must be at one of the category boundaries used by GBD (otherwise the
    returned value will be the prevalence up to the next lowest cutoff).
    `lbwsg_exposure` is LBWSG exposure data from the Artifact
    `cat_df` is the LBWSG category data DataFrame created by Nathaniel's functions
    """
    # get list of low birthweight categories
    low_bw = cat_df.bw_end <= low_bw_cutoff
    low_bw_cats = cat_df.loc[low_bw, 'lbwsg_category']
    # subset LBWSG exposure data to low birthweight categories
    idx = pd.IndexSlice
    low_bw_exposure = lbwsg_exposure.loc[idx[:,:,:,:,:,low_bw_cats]]
    # get groupby columns to sum over categories (in 'parameter' column)
    sum_index_cols = lbwsg_exposure.index.names.difference(['parameter'])
    # sum over low birthweight categories to get overall LBW exposure
    low_bw_prevalence = low_bw_exposure.groupby(sum_index_cols).sum()
    return low_bw_prevalence

# From 2022_02_03a_v5.3.2_wasting_stunting_vs_time_by_scenario
def apply_function(df, func):
    new_df = (
        df.pipe(vp.value)
        .pipe(func)
        .reset_index()
        .assign(measure=lambda df: df['measure'].map(lambda measure_string: f"{func.__name__}({measure_string})"))
    )
    return new_df

def reciprocal(x):
    return 1/x

apply_function(wasting_prevalence, np.log)
apply_function(wasting_prevalence, reciprocal)
