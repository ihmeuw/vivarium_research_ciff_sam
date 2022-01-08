import pandas as pd, numpy as np
from scipy.interpolate import griddata, RectBivariateSpline

from get_draws.api import get_draws

# Functions to read in and format LBWSG input data

def cats_to_ordered_categorical(risk_categories: pd.Series) -> pd.Categorical:
    """Converts a Series of risk categories stored as strings of the form 'cat###' to a Pandas Categorical,
    with the natural sort order (https://en.wikipedia.org/wiki/Natural_sort_order).
    The ordered Categorical is useful for automatically aligning data by risk category.
    """
    ordered_categories = sorted(risk_categories.unique(), key=lambda s: int(s.strip('cat')))
    return pd.Categorical(risk_categories, categories=ordered_categories, ordered=True)

def string_to_interval(interval_strings: pd.Series) -> pd.Series:
    """Converts a Series of strings of the form '(a, b)', '[a, b)', '(a, b]', or '[a, b]' into
    a Series of pandas Interval objects. The a's and b's must be nonnegative integers.
    """
    interval_pattern = r'(?P<left_delimiter>[\(\[])(?P<left>\d+), (?P<right>\d+)(?P<right_delimiter>[\)\]])'
    df = interval_strings.str.extract(interval_pattern)
    df['left_closed'] = (df['left_delimiter'] == '[').astype(int) # 1 or 0
    df['right_closed'] = (df['right_delimiter'] == ']').astype(int) # 1 or 0
    df['closed'] = (df['left_closed']+2*df['right_closed']).map({0:'neither', 1:'left', 2:'right', 3:'both'})
    intervals = np.vectorize(pd.Interval)(df['left'].astype(int), df['right'].astype(int), df['closed'])
    return pd.Series(intervals, index=interval_strings.index, name=interval_strings.name)

# `read_cat_df` requires `cats_to_ordered_categorical` and
# `string_to_interval` helper functions
def read_cat_df(filename: str) -> pd.DataFrame:
    """Reads in the LBWSG category data .csv as a DataFrame, and converts the category column into a
    pandas ordered Categorical and the GA and BW interval columns into Series of pandas Interval objects.
    """
    cat_df = pd.read_csv(filename)
    cat_df['lbwsg_category'] = cats_to_ordered_categorical(cat_df['lbwsg_category'])
    cat_df['parameter'] = cat_df['lbwsg_category']
    cat_df['ga_interval'] = string_to_interval(cat_df['ga_interval'])
    cat_df['bw_interval'] = string_to_interval(cat_df['bw_interval'])
    return cat_df

# `get_rr_data` requires `cats_to_ordered_categorical` helper function
def get_rr_data(source='get_draws', rr_key=None, draw=None, preprocess=False) -> pd.DataFrame:
    """Reads GBD's LBWSG relative risk data from an HDF store or DataFrame or pulls it using get_draws,
    and, if preprocess is True, reformats the RRs into a DataFrame containing a single RR value for
    each age group, sex, and category.
    The DataFrame is indexed by age_group_id and sex_id, and the columns are the LBWSG categories.
    The single RR value will be from the specified draw, or the mean of all draws if draw=='mean'.
    If preprocess is False, the raw GBD data will be returned instead.
    """
    if isinstance(source, pd.DataFrame):
        # Assume source is raw rr data from GBD
        rr = source
    elif source == 'get_draws':
        # Call get draws
        LBWSG_REI_ID = 339
        DIARRHEAL_DISEASES_CAUSE_ID = 302 # Can be any most-detailed cause affected by LBWSG
        GLOBAL_LOCATION_ID = 1 # Passing any location will return RRs for Global
        GBD_2019_ROUND_ID = 6
        rr = get_draws(
          gbd_id_type=('rei_id','cause_id'),
          gbd_id=(LBWSG_REI_ID, DIARRHEAL_DISEASES_CAUSE_ID),
          source='rr',
          location_id=GLOBAL_LOCATION_ID,
          year_id=2019,
          gbd_round_id=GBD_2019_ROUND_ID,
          status='best',
          decomp_step='step4',
        )
    else:
        # Assume source is a string representing a filepath, a Path object,
        # or an HDFStore object. Will raise an error if rr_key is None and
        # source hdf contains more than one pandas object.
        rr = pd.read_hdf(source, rr_key)

    if preprocess:
        draw_cols = rr.filter(like='draw').columns
        rr = rr.assign(lbwsg_category=lambda df: cats_to_ordered_categorical(df['parameter'])) \
             .set_index(['age_group_id', 'sex_id', 'lbwsg_category'])[draw_cols]

        if draw is None:
            raise ValueError("draw must be specified if preprocess is True")
        elif draw == 'mean':
            rr = rr.mean(axis=1)
        else:
            rr = rr[f'draw_{draw}']
        # After unstacking, each row is one age group and sex, columns are categories
        # Categories will be sorted in natural sort order because they're stored in an ordered Categorical
        rr = rr.unstack('lbwsg_category')
    return rr

def get_tmrel_mask(
    ga_coordinates: np.ndarray, bw_coordinates: np.ndarray, cat_df: pd.DataFrame, grid: bool
) -> np.ndarray:
    """Returns a boolean mask indicating whether each pair of (ga,bw) coordinates is in a TMREL category.

    The calling convention using the `grid` parameter is the same as for the scipy.interpolate classes:

        If grid is True, the 1d arrays ga_coordinates and bw_coordinates are interpreted as lists of
        x-axis and y-axis coordinates defining a 2d grid, i.e. the coordinates to look up are the pairs
        in the Carteian product ga_coordinates x bw_coordinates, and the returned mask will have shape
        (len(ga_coordinates), len(bw_coordinates)).

        If grid is False, the 1d arrays ga_coordinates and bw_coordinates must have the same length and are
        interpreted as listing pairs of coordinates, i.e. the coordinates to look up are the pairs in
        zip(ga_coordinates, bw_coordinates), and the returned mask will have shape (n,), where n is the
        common length of ga_coordinates and bw_coordinates.
    """
    TMREL_CATEGORIES = ('cat53', 'cat54', 'cat55', 'cat56')

    # Set index of cat_df to a MultiIndex of pandas IntervalIndex objects to enable
    # looking up LBWSG categories by (GA,BW) coordinates via DataFrame.reindex
    cat_data_by_interval = cat_df.set_index(['ga_interval', 'bw_interval'])

    # Create a MultiIndex of (GA,BW) coordinates to look up,
    # one row for each interpolation point
    if grid:
        # Interpret GA and BW coordinates as the x and y coordinates of a grid
        # (take Cartesian product)
        ga_bw_coordinates = pd.MultiIndex.from_product(
            (ga_coordinates, bw_coordinates), names=('ga_coordinate', 'bw_coordinate')
        )
    else:
        # Interpret GA and BW coordinates as a sequence of points (zip the coordinate arrays)
        ga_bw_coordinates = pd.MultiIndex.from_arrays(
            (ga_coordinates, bw_coordinates), names=('ga_coordinate', 'bw_coordinate')
        )

    # Create a DataFrame to store category data for each (GA,BW) coordinate in the grid
    ga_bw_cat_data = pd.DataFrame(index=ga_bw_coordinates)

    # Look up category for each (GA,BW) coordinate and check whether it's a TMREL category
    ga_bw_cat_data['lbwsg_category'] = (
      cat_data_by_interval['lbwsg_category'].reindex(ga_bw_coordinates))
    ga_bw_cat_data['in_tmrel'] = ga_bw_cat_data['lbwsg_category'].isin(TMREL_CATEGORIES)

    # Pull the TMREL mask out of the DataFrame and convert to a numpy array,
    # reshaping into a 2D grid if necessary
    tmrel_mask = ga_bw_cat_data['in_tmrel'].to_numpy()
    if grid:
        # Make a 2D mask the same shape as the grid,
        tmrel_mask = tmrel_mask.reshape((len(ga_coordinates), len(bw_coordinates)))
    return tmrel_mask

def make_lbwsg_log_rr_interpolator(rr: pd.DataFrame, cat_df: pd.DataFrame) -> pd.Series:
    """Returns a length-4 Series of RectBivariateSpline interpolators for the logarithms of
    the given set of LBWSG RRs, indexed by age_group_id and sex_id.
    """
    # Step 1: Get coordinates of LBWSG category midpoints, indexed by category
    # Category index will be in natural sort order
    interval_data_by_cat = cat_df.set_index('lbwsg_category')
    ga_midpoints = interval_data_by_cat['ga_midpoint']
    bw_midpoints = interval_data_by_cat['bw_midpoint']

    # Step 2: Take logs of LBWSG relative risks
    # Each row of RR is one age group and sex, columns are LBWSG categories
    # Categories (columns) are in natural sort order because they're stored
    # in an ordered Categorical
    log_rr = np.log(rr)

    # Make sure z values are correctly aligned with x and y values
    # (should hold because categories are ordered)
    assert ga_midpoints.index.equals(log_rr.columns)\
    and bw_midpoints.index.equals(log_rr.columns),\
    "Interpolation (ga,bw)-points and rr-values are misaligned!"

    # Step 3: Define intermediate grid $G$ for nearest neighbor interpolation
    # Intermediate grid G = Category midpoints plus boundary points
    ga_min, bw_min = interval_data_by_cat[['ga_start', 'bw_start']].min()
    ga_max, bw_max = interval_data_by_cat[['ga_end', 'bw_end']].max()

    ga_grid = np.append(np.unique(ga_midpoints), [ga_min, ga_max]); ga_grid.sort()
    bw_grid = np.append(np.unique(bw_midpoints), [bw_min, bw_max]); bw_grid.sort()

    # Steps 4 and 5a: Create an interpolator for each age_group and sex
    # (4 interpolators total)
    def make_interpolator(log_rr_for_age_sex: pd.Series) -> RectBivariateSpline:
        # Step 4: Use `griddata` to extrapolate to $G$ via nearest neighbor interpolation
        logrr_grid_nearest = griddata(
          (ga_midpoints, bw_midpoints),
          log_rr_for_age_sex,
          (ga_grid[:,None], bw_grid[None,:]),
          method='nearest',
          rescale=True
        )
        # Step 5a: Create a `RectBivariateSpline` object from the extrapolated values on G
        return RectBivariateSpline(ga_grid, bw_grid, logrr_grid_nearest, kx=1, ky=1)

    # Apply make_interpolator function to each of the 4 rows of log_rr
    log_rr_interpolator = log_rr.apply(
    make_interpolator, axis='columns').rename('lbwsg_log_rr_interpolator')
    return log_rr_interpolator

# Step 5: Interpolate log(RR) to the rectangle [0,42wk]x[0,4500g]
# via bilinear interpolation

# First create a test population to which we'll assign relative risks
def generate_uniformly_random_population(pop_size, seed=12345):
    """Generate a uniformly random test population of size pop_size, with attribute columns
    'age_group_id', 'sex_id', 'gestational_age', 'birthweight'.
    """
    rng=np.random.default_rng(seed)
    pop = pd.DataFrame(
        {
            'age_group_id': rng.choice([2,3], size=pop_size),
            'sex_id': rng.choice([1,2], size=pop_size),
            'gestational_age': rng.uniform(0,42, size=pop_size),
            'birthweight': rng.uniform(0,4500, size=pop_size),
        }
    ).rename_axis(index='simulant_id')
    return pop

# Step 5b: Interpolate log(RR) to (GA,BW) coordinates for a simulated population

def interpolate_lbwsg_rr_for_population(
    pop: pd.DataFrame, log_rr_interpolator: pd.Series, cat_df: pd.DataFrame) -> pd.Series:
    """Return the interpolated RR for each simulant in a population."""
    # Initialize log(RR) to 0, and mask out points in TMREL when we interpolate (Step 7)
    logrr_for_pop = pd.Series(0, index=pop.index, dtype=float)
    tmrel = get_tmrel_mask(pop['gestational_age'], pop['birthweight'], cat_df, grid=False)

    # Step 5b: Interpolate log(RR) to (GA,BW) coordinates for a simulated population
    for age, sex in log_rr_interpolator.index:
        to_interpolate = (pop['age_group_id']==age) & (pop['sex_id']==sex) & (~tmrel)
        subpop = pop.loc[to_interpolate]
        logrr_for_pop.loc[to_interpolate] = log_rr_interpolator[age, sex](
          subpop['gestational_age'], subpop['birthweight'], grid=False)

    # Step 6: Exponentiate to recover the relative risks
    rr_for_pop = np.exp(logrr_for_pop).rename("lbwsg_relative_risk")
    return rr_for_pop
