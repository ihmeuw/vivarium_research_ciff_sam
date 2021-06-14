import numpy as np, pandas as pd
import sympy as sym
from sympy import symbols, Matrix, solve, simplify

from db_queries import get_outputs as go, get_ids
from vivarium import Artifact

## PULL DATA

# we solved in terms of arbitrary ds, ps, and fs, but for a specific age_group/sex, we can plug these in from GBD.
# pulling that data here:

#exposure stratified mortality rates
mort = pd.read_csv('/ihme/homes/beatrixh/vivarium_research_ciff/wasting_transitions/cat_strat_mort_2021_06_08.csv')

# wasting exposures

#pulling from artifact - how to pull centrally from GBD? 
art = Artifact('/ihme/costeffectiveness/artifacts/vivarium_ciff_sam/ethiopia.hdf', filter_terms=['year_start == 2019', 'age_start <  0.076712', f'age_end <= 5'])
art_wasting_exp = art.load('risk_factor.child_wasting.exposure').reset_index()

art_wasting_exp['mean_value'] = art_wasting_exp.iloc[:,['draw' in i for i in art_wasting_exp.columns]].mean(axis=1)
wasting_exp = art_wasting_exp[['sex', 'age_start', 'age_end', 'year_start', 'year_end', 'parameter', 'mean_value']]

wasting_exp['sex_id'] = np.where(wasting_exp['sex'] == 'Male', 1, 2)
wasting_exp['age_group_id'] = np.where(wasting_exp['age_start'] == 0, 4, 5) 

# pull acmr
acmr_df = go(
    "cause", 
    cause_id=294, #all causes
    location_id=179, 
    metric_id=3, 
    year_id=2019, 
    age_group_id=[4,5], 
    measure_id=1, 
    sex_id=[1,2,3], 
    gbd_round_id = 6,
    decomp_step='step5',
    version='latest',
)

acmr_df = acmr_df[
    ['cause_id','cause_name','age_group_id',
     'metric_name','sex_id','val','upper','lower']
].sort_values(
    ['metric_name','cause_id','cause_name','sex_id','age_group_id'])

def set_ds(sex_id, age_group_id):
    d1 = float(mort.loc[(mort.sex_id==sex_id) &
                        (mort.age_group_id==age_group_id) &
                        (mort.cat=='cat1')].mortality_hazard)
    d2 = float(mort.loc[(mort.sex_id==sex_id) &
                        (mort.age_group_id==age_group_id) &
                        (mort.cat=='cat2')].mortality_hazard)
    d3 = float(mort.loc[(mort.sex_id==sex_id) &
                        (mort.age_group_id==age_group_id) &
                        (mort.cat=='cat3')].mortality_hazard)
    d4 = float(mort.loc[(mort.sex_id==sex_id) &
                        (mort.age_group_id==age_group_id) &
                        (mort.cat=='cat4')].mortality_hazard)
    return d1, d2, d3, d4
    


def set_fs(sex_id, age_group_id):
    # pull prev (for emr)
    f1 = float(wasting_exp.loc[
        (wasting_exp.age_group_id==4) &
        (wasting_exp.sex_id==2) &
        (wasting_exp.parameter=='cat1')].mean_value)
    f2 = float(wasting_exp.loc[
        (wasting_exp.age_group_id==4) &
        (wasting_exp.sex_id==2) &
        (wasting_exp.parameter=='cat2')].mean_value)
    f3 = float(wasting_exp.loc[
        (wasting_exp.age_group_id==4) &
        (wasting_exp.sex_id==2) &
        (wasting_exp.parameter=='cat3')].mean_value)
    f4 = float(wasting_exp.loc[
        (wasting_exp.age_group_id==4) &
        (wasting_exp.sex_id==2) &
        (wasting_exp.parameter=='cat4')].mean_value)
    
    return f1, f2, f3, f4


def set_ps(sex_id, age_group_id, time_step):
    acmr = float(acmr_df.loc[(acmr_df.age_group_id==4) &
                             (acmr_df.sex_id==2)].val)
    p0 = 1 - np.exp(-acmr*time_step/365)
    Z = 1 + p0 #normalize prevalences of wasting exposures by reincarnation pool prev
    
    f1, f2, f3, f4 = set_fs(sex_id, age_group_id)

    return p0, f1/Z, f2/Z, f3/Z, f4/Z




## SOLVE EQNS
s4, i3 = symbols('s4 i3')
r4, s3, i2 = symbols('r4 s3 i2')
r3, s2, i1 = symbols('r3 s2 i1')
r2, s1 = symbols('r2 s1')
d4, d3, d2, d1, sld, dother = symbols('d4 d3 d2 d1 sld dother')
f4, f3, f2, f1 = symbols('f4 f3 f2 f1')

p4, p3, p2, p1, p0 = symbols('p4 p3 p2 p1 p0')

acmr = sym.Symbol('acmr')

dur_cat3 = sym.Symbol('dur_cat3')

unknowns = [s1,s2,s3,s4,r2,r3,r4,i1,i2,i3,d1,d2,d3,d4,f1,f2,f3,f4]

def add_eq(terms, y, i, A, v):
    """
    For input equation y = sum([coeff*var for var:coeff in {terms}])
    adds right side of equation to to row i of matrix A

    adds y to row i of vector v
    """
    for x in terms.keys():
        A[x][i] = terms[x]
    v.iloc[i] = y


def build_matrix(eqns, unknowns):
    """
    INPUT
    ----
    eqns: a list of sympy equations
    unknowns: a list of sympy unknowns
    ----
    OUTPUT
    ----
    A:  a matrix containing the coefficients of LHS of all eq in eqns.
      nrows = number of equations
      rcols = number of unknowns
    b: an nx1 matrix containing the RHS of all the eqns
    x: a sympy matrix of the unknowns
    """
    n_eqns = len(eqns)
    n_unknowns = len(unknowns)

    # frame for matrix/LHS equations.
    # nrows = n_eqns, ncols = n_unknowns
    A = pd.DataFrame(
        index = range(n_eqns),
        columns = unknowns,
        data = np.zeros([n_eqns,n_unknowns])
    )

    # frame for RHS of equations
    b = pd.DataFrame(index = range(n_eqns), columns = ['val'])

    # populate LHS/RHS
    i = 0
    for eq in eqns:

        add_eq(eq[0], eq[1], i, A, b)
        i += 1

    # convert to sympy matrices
    A = sym.Matrix(A)
    b = sym.Matrix(b)
    x = sym.Matrix(unknowns) #vars to solve for

    return A, x, b

