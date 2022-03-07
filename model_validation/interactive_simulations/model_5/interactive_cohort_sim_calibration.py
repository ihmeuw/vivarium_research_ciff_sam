from datetime import datetime, timedelta
from pathlib import Path
import itertools
from vivarium import Artifact, InteractiveContext
import ipywidgets
import pandas as pd, numpy as np, os

def run_cohort_simulation(input_draw,
                          random_seed,
                          x_factor_effect,
                          mild_child_wasting_untreated_recovery_time):
    cohort_baseline = pd.DataFrame()
    cohort_endline = pd.DataFrame()
    controls_endline = pd.DataFrame()
    path = Path('/ihme/homes/alibow/vivarium_ciff_sam/src/vivarium_ciff_sam/model_specifications/ciff_sam.yaml')
    sim = InteractiveContext(path, setup=False)
    sim.configuration.update({'input_data':
                                  {'input_draw_number': input_draw},
                              'randomness':
                                  {'random_seed': random_seed},
                              'population':
                                  {'population_size': population_size},

                              'effect_of_x_factor_on_mild_child_wasting':
                                  {'incidence_rate':
                                       {'relative_risk': x_factor_effect}},
                              'effect_of_x_factor_on_mild_child_wasting_to_moderate_acute_malnutrition':
                                  {'transition_rate':
                                       {'relative_risk': x_factor_effect}},
                              'effect_of_x_factor_on_moderate_acute_malnutrition_to_severe_acute_malnutrition':
                                  {'transition_rate':
                                       {'relative_risk': x_factor_effect}},
                              'child_wasting':
                                  {'mild_child_wasting_untreated_recovery_time': mild_child_wasting_untreated_recovery_time}
                              })
    sim.setup()

    cohort_start_recruiting_time = pd.Timestamp(year=2023, month=1, day=1)
    cohort_recruitment_duration_in_timesteps = 365
    observation_duration_in_days = 365

    sim.run_until(cohort_start_recruiting_time)
    pop_t = sim.get_population()

    controls_baseline = pop_t.loc[(pop_t.child_wasting.isin(['susceptible_to_child_wasting',
                                                             'mild_child_wasting']))
                                  & (pop_t.age > 0.5)]

    for step in list(range(0, cohort_recruitment_duration_in_timesteps + observation_duration_in_days * 2)):

        sim.step()
        pop_t_minus_1 = pop_t.copy()
        pop_t = sim.get_population()
        pop_t['status'] = np.nan

        if step <= cohort_recruitment_duration_in_timesteps:
            # add simulants to the cohort_baseline dataframe upon recovery from SAM or MAM
            # note nature of their recovery as "status"
            # record date of recovery so that they can be followed for a fixed time post-recovery
            for i in [i for i in pop_t.index
                      if i in pop_t_minus_1.index
                         and i not in cohort_baseline.index]:

                if ((pop_t_minus_1['child_wasting'].loc[i] == 'moderate_acute_malnutrition')
                        & (pop_t['child_wasting'].loc[i] == 'mild_child_wasting')):

                    pop_t.loc[i, 'status'] = 'recovered_from_mam'
                    cohort_baseline = cohort_baseline.append(
                        pd.DataFrame(pop_t.loc[i]).transpose().rename(columns={'mild_child_wasting_event_time':
                                                                                   'cohort_baseline_entrance_time'}))

                elif pop_t_minus_1['child_wasting'].loc[i] == 'severe_acute_malnutrition':

                    if pop_t['child_wasting'].loc[i] == 'mild_child_wasting':
                        pop_t.loc[i, 'status'] = 'recovered_from_sam'
                        cohort_baseline = cohort_baseline.append(
                            pd.DataFrame(pop_t.loc[i]).transpose().rename(columns={'mild_child_wasting_event_time':
                                                                                       'cohort_baseline_entrance_time'}))
                    elif pop_t['child_wasting'].loc[i] == 'moderate_acute_malnutrition':
                        pop_t.loc[i, 'status'] = 'defaulted_from_sam_to_mam'
                        cohort_baseline = cohort_baseline.append(
                            pd.DataFrame(pop_t.loc[i]).transpose().rename(
                                columns={'moderate_acute_malnutrition_event_time':
                                             'cohort_baseline_entrance_time'}))
        # if it has been one year past entering cohort and they are still <5 years of age (tracked), add them to the endline data frame
        # record their x-factor and wasting treatment exposures as well as MAM/SAM event counts at one year
        if step >= observation_duration_in_days * 2:
            for i in [i for i in cohort_baseline.index if i in pop_t.index and i not in cohort_endline.index]:

                if ((sim._clock.time - cohort_baseline.loc[i, 'cohort_baseline_entrance_time'])
                        > np.timedelta64(observation_duration_in_days, 'D')):
                    cohort_endline = cohort_endline.append(pd.DataFrame(pop_t.loc[i]).transpose())
                    cohort_endline.loc[i, 'x_factor_exposure'] = sim.get_value('x_factor.exposure')(pop_t.index).loc[i]
                    cohort_endline.loc[i, 'wasting_treatment_exposure'] = \
                    sim.get_value('wasting_treatment.exposure')(pop_t.index).loc[i]

        # record data for healthy controls one year after baseline data
        if step == observation_duration_in_days * 2:
            controls_endline = pop_t.loc[[i for i in controls_baseline.index if i in pop_t.index]]
            controls_endline['x_factor_exposure'] = sim.get_value('x_factor.exposure')(pop_t.index).loc[
                controls_endline.index]
#            controls_endline['wasting_treatment_exposure'] = \
#            sim.get_value('wasting_treatment.exposure')(pop_t.index).loc[controls_endline.index]

    return cohort_baseline, cohort_endline, controls_baseline, controls_endline


def format_and_save_results(outputdir,
                            cohort_baseline,
                            cohort_endline,
                            controls_baseline,
                            controls_endline,
                            input_draw,
                            random_seed,
                            x_factor_effect,
                            mild_child_wasting_untreated_recovery_time):
    cohort_baseline['timepoint'] = 'baseline'
    cohort_endline['timepoint'] = 'endline'
    cohort = pd.concat([cohort_baseline, cohort_endline])
    cohort['population'] = 'cohort'

    controls_baseline['timepoint'] = 'baseline'
    controls_endline['timepoint'] = 'endline'
    controls = pd.concat([controls_baseline, controls_endline])
    controls['population'] = 'controls'

    data = pd.concat([cohort, controls])
    data['input_draw'] = input_draw
    data['random_seed'] = random_seed
    data['x_factor_effect'] = x_factor_effect
    data['mild_child_wasting_untreated_recovery_time'] = mild_child_wasting_untreated_recovery_time

    subdir = f'draw_{input_draw}_seed_{random_seed}_x_factor_{x_factor_effect}_mild_transition_{mild_child_wasting_untreated_recovery_time}'
    os.makedirs(outputdir + subdir)
    data.to_csv(outputdir + subdir + '/data.csv')

def main_function(input_draw,
                  random_seed,
                  x_factor_effect,
                  mild_child_wasting_untreated_recovery_time,
                  outputdir):

    (cohort_baseline,
     cohort_endline,
     controls_baseline,
     controls_endline) = run_cohort_simulation(
                                              input_draw,
                                              random_seed,
                                              x_factor_effect,
                                              mild_child_wasting_untreated_recovery_time)
    format_and_save_results(outputdir,
                            cohort_baseline, cohort_endline, controls_baseline, controls_endline,
                            input_draw, random_seed, x_factor_effect, mild_child_wasting_untreated_recovery_time)

if __name__ == "__main__":
    args = sys.argv[1:]
    sim_config, outputdir, input_draw, random_seed, x_factor_effect, mild_child_wasting_untreated_recovery_time = args
    main_function(input_draw, random_seed, x_factor_effect, mild_child_wasting_untreated_recovery_time, outputdir)
