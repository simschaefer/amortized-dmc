
import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import bayesflow as bf
import keras
from dmc import dmc_helpers
import pandas as pd

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

# Load Empirical Data 

narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')

wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')


for network_name in network_names:

    # load specific approximator
    approximator = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name + '.keras')

    # apply approximator on narrow data
    samples_narrow = dmc_helpers.fit_empirical_data(narrow_data, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

    samples_narrow["spacing"]="narrow"

    # apply approximator on wide data
    samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

    samples_wide["spacing"]="wide"

    # combine data sets
    samples_complete = pd.concat((samples_wide, samples_narrow))


    data_path = parent_dir + '/data/empirical_estimates/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save samples
    samples_complete.to_csv(data_path + network_name + '.csv')

    print(network_name, 'estimation completed')





