import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle
import bayesflow as bf
import keras
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from dmc import dmc_helpers

# get parent directory:
scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
# check parent directory (should be '.../amortized-dmc/')
print(f'parent_dir: {parent_dir}', flush=True)


# Priors were updated using initial priors with estimated/fixed sdr:
#network_name = 'initial_priors_sdr_fixed'
network_name = 'initial_priors_sdr_estimated'

# load model specs
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# load approximator
approximator = keras.saving.load_model(parent_dir +"/training_checkpoints/" + network_name + ".keras")

# load empirical data
narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')

# participant IDs that should be used for prior updating (randomly sampled):
train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214])

# select only training participants
train_data_narrow = narrow_data[narrow_data['participant'].isin(train_idx)]

## check rts and accuracies

plt.figure()
sns.kdeplot(train_data_narrow, x='rt', hue='congruency_num')

plt.figure()
sns.histplot(train_data_narrow.groupby(['participant', 'congruency_num']).mean('accuracy').reset_index(), x='accuracy', hue='congruency_num')

# posterior samples 
empirical_samples_narrow = dmc_helpers.fit_empirical_data(train_data_narrow, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

# summary stats of the resulting posteriors -> updated priors
updated_priors_narrow = empirical_samples_narrow.agg(['mean', 'std'])


