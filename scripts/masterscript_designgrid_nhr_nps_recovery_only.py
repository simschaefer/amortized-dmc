
# Import Libraries
import os
import bayesflow as bf
import numpy as np
import math
import os
import pickle
import sys
import pandas as pd
import seaborn as sns
import time
import tensorflow as tf
from bayesflow import computational_utilities as utils
from numba import njit
import re
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from bayesflow.computational_utilities import maximum_mean_discrepancy
from keras.utils import to_categorical

## read design grid

# print(f"TF Version: {tf.__version__}")
#
# devices = tf.config.list_physical_devices()
# print("\nDevices: ", devices)
#
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     details = tf.config.experimental.get_device_details(gpus[0])
#     print("GPU details: ", details)

myhost = os.uname()[1]

RNG = np.random.default_rng(2023)
model_start = datetime.now()
#%%

if myhost != 'psyml201':
    arguments = sys.argv[1:]
    print(f"passed argument: {arguments}")
    index = int(arguments[1])
    design_grid_num = int(arguments[2])
    slurm_id = str(arguments[0])
    model_title = "dmc_"+ str(arguments[0]) + 'design_grid' + str(design_grid_num)+ '_condition' + str(index)
    design_grid_dir = '/data/design_grid' + str(design_grid_num) + '.csv'
else:
    index = 1149
    design_grid_num = 6
    slurm_id = 'PC'
    model_title = "dmc_726534design_grid6_condition1149"
    design_grid_num = 6
    # model_title = 'dmc_418664design_grid6_condition54'


    design_grid_dir = '/data/design_grid' + str(design_grid_num) + '.csv'

model_info = {
    'model_title': model_title,
    'link_function': None, # choose 'normal cdf' to transform normals to uniform priors
    'normal_restriction': 'positive only',
    'model': 'dmc',
    'log_file': 'dmc_log_file.csv',
    'slurm_id': slurm_id,
    'write_log': True,
    'load_pretrained': True,
    'save_plots': True,
    'compute_mmd': True,
    'plot_mms': False,
    'plot_dataspace': True,
    'save_rt_data': False,
    'x0_var': 'trial', # or 'fixed' -> specify X0_beta_shape
    'a_var': 'estimated', # or 'fixed'
    'a_value': None,
    'n_epochs': 100,
    'n_iterations': 1000,
    'min_obs': 50,
    'max_obs': 2000,
    'n_conditions': 2,
    'batch_size': 32,
    'save_posterior_data': True,
    'n_summary_dims': 32,
    'coupling_design': "spline", # oder spline
    'n_coupling_layers': 12,
    'n_validation_sims': 200,
    'early_stopping': False,
    ## Priors ##
    'X0_beta_shape_fixed': 3,
    'X0_value': 0,
    'tmax': 1200,
    'sigma': 4,
    'dt': 1,
    'n_recovery_simulations': 1000,
    'n_posterior_samples': 10000,
    'simulation_function': 'vectorized',
    'nonconvergents': 'resampled',
    'dir_design_grid': design_grid_dir,
    'start_time': model_start,
    'comment': 'extended x_shape range (Luo & proctor, 2022), a fixed',
    'attention_setting_key_dim': 128,
    'num_heads': 8,
    'num_inducing_points': 32,
    'num_seeds': 4,
    'num_dense_fc': 2,
    'dense_setting_units': 256,
    'dense_setting_activation': 'relu',
    'min_scale_alpha': 1,
    'max_scale_alpha': 1,
    'learning_rate': 5e-4,
    'n_repetitions_mmd': 100}



if myhost == 'psyml201':
    script_dir = os.getcwd()

        # modify parent distribution (on Mogon)
    if script_dir == '/home/schasimo':
        parent_dir = script_dir
    else:
        parent_dir = os.path.dirname(script_dir)
else:

    parent_dir = os.getcwd() + '/BF-DMC'


if model_info['load_pretrained']:
    # model titles
    list_files = os.listdir(parent_dir + '/plots')
    list_files =  [path for path in list_files if 'design_grid6' in path]
    model_title = [path for path in list_files if 'condition'+ str(index) in path]
    model_title.sort()
    model_title = model_title[0]
    model_info['model_title'] = model_title


if myhost == 'psyml201':
    # model_info['load_pretrained'] = False
    # model_info['save_plots'] = False
    model_info['plot_mms'] = False
    model_info['compute_mmd'] = False
    model_info['plot_dataspace'] = False

    def find_directories(mkdir = True):

        script_dir = os.getcwd()

        # modify parent distribution (on Mogon)
        if script_dir == '/home/schasimo':
            parent_dir = script_dir
        else:
            parent_dir = os.path.dirname(script_dir)

        model_dir = parent_dir+"/networks/"+ model_title+"/"
        # Get the parent directory (one level above)
        functions_dir = parent_dir+"/functions"
        model_data_dir = parent_dir + "/data/model_data/"

        # Add the script's directory to the Python path
        sys.path.append(functions_dir)

        dirs = ['/plots/', '/data/posteriors/', '/data/summary_statistics/', '/networks/']

        if mkdir:
            for dir in dirs:
                path = parent_dir + dir + model_title
                if not os.path.exists(path):
                    os.makedirs(path)

        return script_dir, parent_dir, functions_dir, model_data_dir, model_dir
else:
    def find_directories(mkdir = True):

        script_dir = os.getcwd() + '/BF-DMC/models'

        # modify parent distribution (on Mogon)
        parent_dir = os.getcwd() + '/BF-DMC'


        model_dir = parent_dir+"/networks/"+ model_title+"/"
        # Get the parent directory (one level above)
        functions_dir = parent_dir+"/functions"
        model_data_dir = parent_dir + "/data/model_data/"

        # Add the script's directory to the Python path
        sys.path.append(functions_dir)

        dirs = ['/plots/', '/data/posteriors/', '/networks/']

        if mkdir:
            for dir in dirs:
                path = parent_dir + dir + model_title
                if not os.path.exists(path):
                    os.makedirs(path)

        return script_dir, parent_dir, functions_dir, model_data_dir, model_dir


script_dir, parent_dir, functions_dir, model_data_dir, model_dir = find_directories(mkdir = False)

figure_path = parent_dir+"/plots/" + model_title + "/" + model_title



import dmc_functions as dmc

print('script_dir = '+script_dir)
print('parent_dir = '+parent_dir)
print('functions_dir = '+functions_dir)
print('model_data_dir = '+model_data_dir)
print('model_dir = '+model_dir)
print('design_grid_dir = '+design_grid_dir)

## read design grid

design_grid = pd.read_csv(parent_dir+model_info['dir_design_grid'], index_col=0)


# if myhost != 'psyml201':
#     # write slurm id into design grid
#     design_grid['slurm_id'][index] = slurm_id
#     design_grid.to_csv(parent_dir+model_info['dir_design_grid'])

# replace NaN with None
design_grid.replace({np.nan: None})

condition = design_grid.iloc[index]

condition.to_dict()

model_info.update(condition)

sys.path.append(parent_dir)

# parameters estimated
param_names = ['A', 'tau', 'mu_c', 'mu_r', 'b']
PARAM_NAMES = [r'$A$', r'$\tau$', r'$\mu_c$',r'$\mu_r$', r'$b$']

# append parameter names if needed
if model_info['sd_r_var'] == 'estimated':
    param_names.append('sd_r')
    PARAM_NAMES.append(r'$sd_r$')

if model_info['a_var'] == 'estimated':
    param_names.extend('a')
    PARAM_NAMES.append(r'$a$')

n_pars = len(param_names)

families_list = [name + '_prior' for name in param_names]

prior_families = [model_info[key] for key in families_list if key in model_info]

restriction = model_info['normal_restriction']
link_fun = model_info['link_function']

## extract prior parameters from model_info

prior_par_names = []
prior_pars_arr = np.zeros((n_pars, 2))
# prior_pars_arr = np.concatenate((prior_pars_arr, np.array(param_names).reshape(n_pars,1)), axis = 1)

for i in range(n_pars):
    if prior_families[i] == 'gamma':
        suffix = ['shape', 'rate']
    elif prior_families[i] == 'normal':
        suffix = ['mean', 'sd']

    for j in range(len(suffix)):
        par_name = param_names[i] + '_prior_' + suffix[j]
        prior_par_names.append(par_name)
        prior_pars_arr[i, j] = model_info[par_name]

prior_fun = partial(dmc.prior_nps,
                    families = np.array(prior_families),
                    pars1 = prior_pars_arr[:,0],
                    pars2 = prior_pars_arr[:,1],
                    param_names = np.array(param_names),
                    restriction = restriction)

# prior_batchable_fun = partial(dmc.alpha_gen,
#                         min = model_info['min_scale_alpha'],
#                         max = model_info['max_scale_alpha'],
#                         size = n_pars)

# prior_fun(prior_batchable_fun())
# context_alpha = partial(dmc.alpha_gen,
#                         min = 1,
#                         max = 1,
#                         size = n_pars)
# single test prior function

# prior_context = bf.simulation.ContextGenerator(batchable_context_fun=prior_batchable_fun)

prior = bf.simulation.Prior(prior_fun = prior_fun,
                            #context_generator = prior_context,
                            param_names = PARAM_NAMES)



# prior means and sds for standardizing
prior_means, prior_stds = prior.estimate_means_and_stds()

# # save prior plots
# if model_info['save_plots']:
#     fig = f.get_figure()
#     fig.savefig(parent_dir+"/plots/" + model_title + "/" + model_title +"_priors.png")


## test functions
theta = prior_fun()

dmc.trial(theta[0],
          theta,
          tmax = model_info['tmax'],
          sigma = model_info['sigma'],
          dt = model_info['dt'],
          sd_r_var = model_info['sd_r_var'],
          a_var = model_info['a_var'],
          x0_var = model_info['x0_var'],
          a_value = model_info['a_value'],
          X0_value = model_info['X0_value'],
          X0_beta_shape_fixed = model_info['X0_beta_shape_fixed'])

sim_batchable_fun = partial(dmc.condition_matrix, n_conditions = model_info['n_conditions'])

sim_non_batchable_fun = partial(dmc.random_n_obs, min_obs = model_info['min_obs'], max_obs = model_info['max_obs'])

# test function:
# dmc.experiment(theta,
#                sim_batchable_fun(1000),
#                sim_non_batchable_fun(),
#                    tmax = model_info['tmax'],
#                    sigma = model_info['sigma'],
#                    dt = model_info['dt'],
#                    sd_r_var = model_info['sd_r_var'],
#                    a_var = model_info['a_var'],
#                    x0_var = model_info['x0_var'],
#                    a_value = model_info['a_value'],
#                    X0_value = model_info['X0_value'],
#                    X0_beta_shape_fixed = model_info['X0_beta_shape_fixed'])

experiment_fun = partial(dmc.experiment,
                         tmax = model_info['tmax'],
                         sigma = model_info['sigma'],
                         dt = model_info['dt'],
                         sd_r_var = model_info['sd_r_var'],
                         a_var = model_info['a_var'],
                         x0_var = model_info['x0_var'],
                         a_value = model_info['a_value'],
                         X0_value = model_info['X0_value'],
                         X0_beta_shape_fixed = model_info['X0_beta_shape_fixed'])

sim_context = bf.simulation.ContextGenerator(
    non_batchable_context_fun=sim_non_batchable_fun,
    batchable_context_fun=sim_batchable_fun,
    use_non_batchable_for_batchable=True)

simulator = bf.simulation.Simulator(
    simulator_fun=experiment_fun,
    context_generator=sim_context)

model = bf.simulation.GenerativeModel(
    prior=prior,
    simulator=simulator,
    name='DMC')

# Simulate some Datasets for sanity checks

model.simulator.context_gen.non_batchable_context_fun = lambda: model_info['max_obs']
#
# sim_out = model(batch_size=100)

# fig_rt, fig_rt_total = dmc.plot_rt(sim_out)

# fig_acc = dmc.plot_acc(sim_out)

# if model_info['save_plots']:
#     fig_rt.savefig(figure_path + "_RTdist_single.png")
#     fig_rt_total.savefig(figure_path + "_RTdist_total.png")
#     fig_acc.savefig(figure_path + "_Accuracy.png")

model.simulator.context_gen.non_batchable_context_fun = sim_non_batchable_fun

config = partial(dmc.configurator_nps,
                 prior_means = prior_means,
                 prior_stds = prior_stds)

# test = config(model(10))

#################  Defining Neural Approximator #################

summary_net = bf.networks.SetTransformer(input_dim=3,
                                         attention_settings=dict(key_dim=model_info['attention_setting_key_dim'],
                                                                 num_heads=model_info['num_heads'],
                                                                 dropout=0.0),
                                         num_inducing_points=model_info['num_inducing_points'],
                                         use_layer_norm=model_info['use_layer_norm'],
                                         num_seeds=model_info['num_seeds'],
                                         dense_settings=dict(units=model_info['dense_setting_units'],
                                                             activation=model_info['dense_setting_activation']),
                                         num_dense_fc=model_info['num_dense_fc'],
                                         summary_dim=model_info['n_summary_dims'],
                                         name="dmc_summary_"+model_title)

# inference_net = bf.networks.InvertibleNetwork( num_params=len(prior.param_names),
#                                                num_coupling_layers = 12,
#                                                coupling_settings={
#                                                    "dropout_prob": 0.1,
#                                                    'bins':64},
#                                                name="dmc_inference_"+model_title)

inference_net = bf.networks.InvertibleNetwork(
    num_params=len(prior.param_names),
    num_coupling_layers=model_info['n_coupling_layers'],
    coupling_design=model_info['coupling_design'],
    #coupling_settings={ 'mc_dropout' : True, 'dense_args' : dict(units=128, activation='elu') },
    name = "dmc_inference"+model_info['model_title'])

amortizer = bf.amortizers.AmortizedPosterior(inference_net,
                                             summary_net,
                                             name='dmc_amortizer_'+model_title,
                                             summary_loss_fun="MMD")

# Define Trainer
trainer = bf.trainers.Trainer(
    generative_model=model,
    amortizer=amortizer,
    configurator=config,
    checkpoint_path=model_dir,
    memory = True,
    default_lr= model_info['learning_rate'])

# Model Summary
amortizer.summary()

# list files (observed data)
list_files = os.listdir(model_data_dir)
# Filter and select filenames containing "exp"
# exp_files = [filename for filename in list_files if "model_data_hedge" in filename]

# data_set_names = [name.rstrip('.csv') for name in list_files]

### EXPLORE DATA SPACE ####

# if model_info['plot_dataspace']:
#     for i, data_set in enumerate(['acdc_clean_data_flanker.csv']):
#
#         print(f"simulate data space for {data_set}")
#         ## read data
#         data = read_data(data_set, model_data_dir)
#
#         x, x_obs = data_space(data,
#                               trainer,
#                               model,
#                               prior_fun,
#
#                               parent_dir,
#                               model_title,
#                               data_set,
#                               save_plots = model_info['save_plots'])

#%%
################# Train the model ###########################

# if model_info['load_pretrained']:
#     n_epochs = 0
#     n_iterations = 0

# else:
#     n_epochs = model_info['n_epochs']
#     n_iterations = model_info['n_iterations']

# start_time = time.time()


# h = trainer.train_online(epochs=n_epochs, iterations_per_epoch=n_iterations,
#                          batch_size=model_info['batch_size'],
#                          #optimizer = optimizer,
#                          save_checkpoint=True,
#                          early_stopping=model_info['early_stopping'],
#                          validation_sims=model_info['n_validation_sims'])

# end_time = time.time()



# model_info['iterations_completed'] = h['train_losses'].shape[0]
# model_info['fitting_time_sec'] = end_time - start_time


#### Recovery Analysis ####

list_data_frames = list()
list_prior_draws = list()
list_post_means = list()
list_n_trials_simulation = list()
list_odd = list()
list_even = list()

n_sims = model_info['n_recovery_simulations']

trial_nums = [50,100,200, 400,600,800,1000, 2000]

for j in range(0, len(trial_nums)):

    print(f"trial_num: {trial_nums[j]}", flush=True)

    # adjust number of trials in simulator:
    model.simulator.context_gen.non_batchable_context_fun = lambda: trial_nums[j]

    # simulate data
    sim_out = model(batch_size=n_sims)
    sim_data = trainer.configurator(sim_out)

    # assess actual number of trials (should be identical to trial_nums[j])
    n_trials = sim_data['summary_conditions'].shape[1]
    list_n_trials_simulation.append(n_trials)

    # SPLIT HALF RELIABILITY

    # odd trials
    odd_data = sim_data.copy()
    odd_data['summary_conditions'] = odd_data['summary_conditions'][:, ::2]
    post_samples_odd = amortizer.sample(odd_data, n_samples = 1000)
    post_samples_odd_not_z = post_samples_odd * prior_stds + prior_means
    post_samples_odd_means = np.mean(post_samples_odd_not_z, axis = 1)

    # even trials
    even_data = sim_data.copy()
    even_data['summary_conditions'] = even_data['summary_conditions'][:, 1::2]
    post_samples_even = amortizer.sample(even_data, n_samples = 1000)
    post_samples_even_not_z = post_samples_even * prior_stds + prior_means
    post_samples_even_means = np.mean(post_samples_even_not_z, axis = 1)

    list_odd.append(pd.DataFrame(post_samples_odd_means))
    list_even.append(pd.DataFrame(post_samples_even_means))

    # PARAMETER RECOVERY
    # fit model with simulated Data
    post_samples = amortizer.sample(sim_data, n_samples = 1000)

    # unstandardize psoterior samples
    post_samples_not_z = post_samples * prior_stds + prior_means

    # compute posterior means
    post_means = np.mean(post_samples_not_z, axis = 1)

    # save prior draws
    par_data = sim_out['prior_draws']

    df_par = pd.DataFrame(par_data)
    df_par.id = j+1
    list_prior_draws.append(df_par)
    list_post_means.append(pd.DataFrame(post_means))

#data_rt = pd.concat(list_data_frames)
data_priors = pd.concat(list_prior_draws)
data_post_means = pd.concat(list_post_means)
data_odd = pd.concat(list_odd)
data_even = pd.concat(list_even)

n_trials_simulation = [item for item in list_n_trials_simulation for _ in range(n_sims)]
participants = list(range(1,n_sims*len(trial_nums)+1))


data_post_means = data_post_means.set_axis(param_names,axis = 1)
data_post_means['participant'] = participants
data_post_means['n_trials'] = n_trials_simulation
data_post_means['model_title'] = model_title

data_priors = data_priors.set_axis(param_names,axis = 1)
data_priors['participant'] = participants
data_priors['n_trials'] = n_trials_simulation
data_priors['model_title'] = model_title

data_odd = data_odd.set_axis(param_names,axis = 1)
data_odd['participant'] = participants
data_odd['n_trials_total'] = n_trials_simulation
data_odd['n_trials_split'] = np.array(n_trials_simulation)/2
data_odd['split'] = 'odd'
data_odd['model_title'] = model_title

data_even = data_even.set_axis(param_names,axis = 1)
data_even['participant'] = participants
data_even['n_trials_total'] = n_trials_simulation
data_even['n_trials_split'] = np.array(n_trials_simulation)/2
data_even['split'] = 'even'
data_even['model_title'] = model_title

#data_rt['model_title'] = model_title

# join to one data frame
# data_combined = pd.merge(data_rt, data_priors, on='participant', how='left')
# data_combined = pd.merge(data_combined, data_post_means, on='participant', how='left')
#
# data_combined['model_title'] = model_title

# if model_info['save_rt_data']:
#     data_rt.to_csv(parent_dir+'/data/recovery_data/simulation_rt_' + model_title +'.csv')
data_priors.to_csv(parent_dir+'/data/recovery_data/simulation_priors_' + model_title +'.csv')
data_post_means.to_csv(parent_dir+'/data/recovery_data/simulation_posteriors_' + model_title +'.csv')

data_odd.to_csv(parent_dir+'/data/reliability_data/data_odd_' + model_title +'.csv')
data_even.to_csv(parent_dir+'/data/reliability_data/data_even_' + model_title +'.csv')

#data_rt['model_title'] = model_title

# join to one data frame
# data_combined = pd.merge(data_rt, data_priors, on='participant', how='left')
# data_combined = pd.merge(data_combined, data_post_means, on='participant', how='left')
#
# data_combined['model_title'] = model_title

# if model_info['save_rt_data']:
#     data_rt.to_csv(parent_dir+'/data/recovery_data/simulation_rt_' + model_title +'.csv')
data_priors.to_csv(parent_dir+'/data/recovery_data/simulation_priors_' + model_title +'.csv')
data_post_means.to_csv(parent_dir+'/data/recovery_data/simulation_posteriors_' + model_title +'.csv')

print('Recovery and Reliability data stored.', flush = True)