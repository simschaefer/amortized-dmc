
import pickle
import pandas as pd
import os

network_names = [
    'updated_priors_sdr_fixed',
    'updated_priors_sdr_estimated',
    'initial_priors_sdr_fixed',
    'initial_priors_sdr_estimated',
]

# get parent directory:
scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
# check parent directory (should be '.../amortized-dmc/')
print(f'parent_dir: {parent_dir}', flush=True)

from dmc import DMC, dmc_helpers

# load data
narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')

wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat((narrow_data, wide_data))

# Before running this code, scripts/empirical_estimates.py has to be executed

# define number of resimulations per participant
num_resims = 100

lst_data = [] 

for network_name in network_names:

    model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
    with open(model_specs_path, 'rb') as file:
        model_specs = pickle.load(file)

    simulator = DMC(**model_specs['simulation_settings'])

    # read samples (from scripts/empirical_estimates.py)
    df_samples = pd.read_csv(parent_dir + '/data_complete/empirical_estimates/' + network_name + '.csv')

    # list of all participants
    parts = df_samples['participant'].unique()

    for spacing, spacing_num in zip(['narrow', 'wide'], [1,0]):
        
        for part in parts:
            
            # number of trials of the given participant
            num_obs = empirical_data[(empirical_data['spacing_num'] == spacing_num) & (empirical_data['participant'] == part)].shape[0]

            # filter sample data for given participant and narrow spacing
            part_data_samples = df_samples[df_samples["participant"]==part]

            part_data_samples = part_data_samples[part_data_samples["spacing"] == spacing]

            # resimulate data
            data_resimulated = dmc_helpers.resim_data_id(part_data_samples, num_obs=num_obs, simulator=simulator, id=part, param_names=model_specs['simulation_settings']['param_names'], id_name='participant')
            
            # exclude non-convergents
            data_resimulated = data_resimulated[data_resimulated["rt"] != -1]

            # recode congruency
            data_resimulated["condition_label"] = data_resimulated["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

            data_resimulated['model'] = network_name

            data_resimulated['spacing'] = spacing

            # save resimulations in list
            lst_data.append(data_resimulated)

# combine data
df_complete = pd.concat(lst_data)

# save resimulation data
data_path = parent_dir + '/data_complete/ppc_data/'

if not os.path.exists(data_path):
      os.makedirs(data_path)

df_complete.to_csv(data_path + '/ppc_data_raw.csv')
