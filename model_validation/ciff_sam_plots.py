import pandas as pd, numpy as np, matplotlib.pyplot as plt

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