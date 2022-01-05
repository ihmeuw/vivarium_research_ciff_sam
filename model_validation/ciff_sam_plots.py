import pandas as pd, numpy as np, matplotlib.pyplot as plt

import re
from pathlib import Path

import model_validation.vivarium_output_processing as vop
import model_validation.ciff_sam_results as csr

def plot_over_time_by_age(df, ylabel='', title='', ax=None):
    """Plot mean value vs. year for each age, with (lower, upper) uncertainty band."""
    if ax is None:
        ax = plt.gca()
    df = csr.age_to_ordered_categorical(df) # Order the age groups chronologically
    agg = df.groupby(['age', 'year'])['value'].describe(percentiles=[.025, .975])
    ages = agg.index.unique('age')
    for age in ages:
        values = agg.xs(age)
        years = values.index
        ax.plot(years, values['mean'], label=age)
        ax.fill_between(years, values['2.5%'], values['97.5%'], alpha=.1)

    ax.set_xlabel('year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax

def plot_draws_over_time_by_age(df, ylabel='', title='', ax=None):
    """Plot trajectory for each draw and mean over draws vs. year for each age.
    https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
    https://matplotlib.org/stable/gallery/color/color_cycle_default.html
    https://matplotlib.org/stable/tutorials/introductory/customizing.html
    """
    if ax is None:
        ax = plt.gca()
    df = csr.age_to_ordered_categorical(df) # Order the age groups chronologically
# Does the same thing as .pivot:
#     draws_by_age_year = df.set_index(['age', 'year', 'input_draw'])['value'].unstack('input_draw')
#     draws_by_age_year = df.pivot(index=['age', 'year'], columns='input_draw', values='value')
    # Use .stratify to aggregate over sex and any other columns as necessary
    # .stratify puts 'input_draw' and 'scenario' in the index as well as the strata
#     draws_by_age_year = vop.stratify(df, ['age', 'year'], reset_index=False).unstack('input_draw')['value']
    draws_by_age_year = df.groupby(
        ['age', 'year', 'input_draw'], observed=True # observed=True needed to omit unused age categories
    )['value'].mean().unstack('input_draw')
    ages = draws_by_age_year.index.unique('age') # These should be sorted automatically
#     color = iter(plt.cm.tab20(np.linspace(0, 1, len(ages))))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color = iter(prop_cycle.by_key()['color'])
#     print(type(color))
    for age in ages:
        draws_by_year = draws_by_age_year.xs(age)
        years = draws_by_year.index.get_level_values('year')
        c = next(color)
        ax.plot(years, draws_by_year.mean(axis=1), label=f"{age}", c=c)
        for draw in draws_by_year.columns:
            ax.plot(years, draws_by_year[draw], c=c, alpha=.2)

    ax.set_xlabel('year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax

def plot_over_time_by_column(df, colname, ylabel='', title='', uncertainty=True, ax=None):
    """Plot mean value vs. year for each value in the column `colname`,
    optionally with (lower, upper) uncertainty band.
    """
    if ax is None:
        ax = plt.gca()
#     df = cs.age_to_ordered_categorical(df) # Order the age groups chronologically
    df = csr.to_ordered_categoricals(df)
    agg = df.groupby([colname, 'year'])['value'].describe(percentiles=[.025, .975])
    col_vals = agg.index.unique(colname)
    for col_val in col_vals:
        values = agg.xs(col_val)
        years = values.index
        ax.plot(years, values['mean'], label=f"{colname}={col_val}")
        if uncertainty:
            ax.fill_between(years, values['2.5%'], values['97.5%'], alpha=.1)

    ax.set_xlabel('year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax

def plot_over_time_by_column_for_each_wasting_state_and_scenario(
    df, colname, ylabel='', suptitle='', uncertainty=True
):
    """Draw a 4x3 figure with rows indexed by wasting state and columns indexed by scenario,
    calling plot_over_time_by_column() for each subplot.
    """
    # Get ordered lists of wasting states and scenarios in the dataframe
    df = csr.to_ordered_categoricals(df)
    wasting_states = df['wasting_state'].unique().sort_values()
    scenarios = df['scenario'].unique().sort_values()
    fig, axs = plt.subplots(len(wasting_states), len(scenarios), figsize=(16, 16))
    for ws_num, wasting_state in enumerate(wasting_states):
        for s_num, scenario in enumerate(scenarios):
    #         print(scenario, wasting_state)
            plot_over_time_by_column(
                df.query("scenario==@scenario and wasting_state==@wasting_state"),
                colname,
                title=f"{scenario}, {wasting_state}",
                ylabel=ylabel,
                uncertainty=uncertainty,
                ax=axs[ws_num, s_num],
            )
    fig.suptitle(suptitle, fontsize=18)
    fig.tight_layout()
    return fig

def plot_over_time_by_column_for_each_scenario(df, colname, ylabel, suptitle, uncertainty):
    """Draw a 3x1 figure with rows indexed by scenario, calling plot_over_time_by_column()
    for each subplot.
    """
    # Get ordered list of scenarios in the dataframe
    df = csr.to_ordered_categoricals(df)
    scenarios = df['scenario'].unique().sort_values()
    fig, axs = plt.subplots(len(scenarios),1, figsize=(12,6*len(scenarios)))
    for s_num, scenario in enumerate(scenarios):
        plot_over_time_by_column(
            df.query("scenario==@scenario"),
            colname,
            ylabel,
            f"{scenario}",
            uncertainty,
            ax=axs[s_num],
        )
    fig.suptitle(suptitle, fontsize=18)
    fig.tight_layout()
    return fig

# We'll use this function to format the figures' title strings into legitimate file names for saving.
def convert_to_variable_name(string):
    """Converts a string to a valid Python variable.
    Runs of non-word characters (regex matchs \W+) are converted to '_', and '_' is appended to the
    beginning of the string if the string starts with a digit (regex matches ^(?=\d)).
    Solution copied from here:
    https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    return re.sub('\W+|^(?=\d)', '_', string)

def create_output_directories(username, model_name, home=True, j_drive=True, share=False):
    """
    Creates directories for saving plots if they don't already exist,
    and returns a list of the directories.
    """
    project_vv_directory_name = 'ciff_malnutrition/verification_and_validation'
    directories=[]

    if home:
        home_output_dir = f'/ihme/homes/{username}/vivarium_results/{project_vv_directory_name}/{model_name}'
        directories.append(home_output_dir)
    if j_drive:
        j_output_dir = f'/home/j/Project/simulation_science/{project_vv_directory_name}/{model_name}'
        directories.append(j_output_dir)
    if share:
        share_output_dir = f'/share/scratch/users/ndbs/vivarium_results/{project_vv_directory_name}/{model_name}'
        directories.append(share_output_dir)

    # Create the output directories if they don't exist
    # Note from Path.mkdir() documentation:
    #   "If mode is given, it is combined with the process’ umask value to determine the file mode and access flags."
    #
    # I don't know what this notebook process' umask value will be, so I don't know if this will actually result
    # in the correct (most permissive) permissions for the directories...
    for directory in directories:
        Path(directory).mkdir(mode=0o777, parents=True, exist_ok=True)
    return directories

def save_figures(figures: dict, directories):
    """
    Saves the figures to the specified directories as .pdf files.
    `figures` is a dictionary mapping figure names (strings) to matplotlib Figure objects.
        The figure names will be converted to valid filenames using the `convert_to_variable_name` function
        before saving.
    `directories` is a list of directories or a single directory name (str).
    """
    if isinstance(directories, str):
        directories = [directories]
    for figure_name, fig in figures.items():
        for directory in directories:
            fig.savefig(f"{directory}/{convert_to_variable_name(figure_name)}.pdf")
