
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
from tqdm.autonotebook import tqdm
from functools import partial
from bayesflow.default_settings import DEFAULT_KEYS, OPTIMIZER_DEFAULTS, TQDM_MININTERVAL
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
    index = 526 #526
    design_grid_num = 6
    slurm_id = 'PC'
    model_title = "dmc_603210design_grid6_condition526" # dmc_603210design_grid6_condition526
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

    parent_dir = os.getcwd() + '/BF-DMC-NEW'


if model_info['load_pretrained']:
    # model titles
    list_files = os.listdir(parent_dir + '/networks')
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

        script_dir = os.getcwd() + '/BF-DMC-NEW/models'

        # modify parent distribution (on Mogon)
        parent_dir = os.getcwd() + '/BF-DMC-NEW'


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
                                                                 dropout=model_info['dropout'],),
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

#%%

#### RESIMULATION EMPIRICAL DATA

list_pred_data_out = list()
list_pred_data_out = list()

spacings = ['wide', 'narrow']

num_resim = 50

nonconvergent_trials = 0
nonconvergent_trials_cum = 0

num_samples = 1000

for spacing in spacings:

    obs_data = pd.read_csv(parent_dir + f'/data/model_data/experiment_data_{spacing}.csv')
    # obs_data['congruency_num'] = 1 - obs_data['congruency_num']
    dmc.check_congruency(obs_data)

    parts = pd.unique(obs_data['participant'])

    n_parts = parts.shape[0]
    part_count = 0

    with tqdm(total=n_parts, desc=f"Resimulation {spacing} Spacing") as p_bar:
        for j, id in enumerate(parts):

            part_count += 1

            part_data = obs_data[obs_data['participant'] == id]

            dir_cond = np.sqrt(part_data.shape[0])

            n_obs = part_data.shape[0]

            # Generate random index for posterior parameter set selection
            index_set = np.random.choice(np.arange(1000), size=num_resim)

            # Get context of simulated data sets (congruency)
            context = part_data.values[np.newaxis:,3].astype('float64')

            # Re-simulate
            pred_data = np.zeros((num_resim, n_obs, 3))

            dir_cond = np.sqrt(part_data.shape[0])

            # print('estimate posteriors')
            part_data_config = {'summary_conditions': part_data.values[np.newaxis,:,1:4].astype('float64'),
                                'direct_conditions': np.sqrt(part_data.shape[0]).reshape(1,1)}

            # print('Start posterior estimation', flush=True)
            posteriors_experiment = amortizer.sample(part_data_config, n_samples=1000)

            start_time = time.time()
            post_samples = amortizer.sample(part_data_config, n_samples = num_samples)
            end_time = time.time()

            time_diff = end_time - start_time

            post_samples_not_z = posteriors_experiment * prior_stds + prior_means

            for i, idx in enumerate(index_set):
                a = post_samples_not_z[idx, -1]

                if model_info['a_prior'] == 'gamma':
                    while a <= 1 or idx in index_set:
                        idx = np.random.choice(np.arange(num_samples), size=1)
                        a = post_samples_not_z[idx, -1]

                if a <= 1:
                    print(f'WARNING: a IS SMALLER OR EQUAL 1: {a}')

                exp_data = experiment_fun(
                    np.array(post_samples_not_z[idx, :]).flatten(),
                    context,
                    np.array(context).shape[0]
                )
                pred_data[i, :, :] = np.concatenate(
                    (exp_data, np.array(context).reshape((np.array(context).shape[0], 1))),
                    axis=1
                )

            nonconvergent_trials = np.sum(pred_data == -1) / 2
            nonconvergent_trials_cum += nonconvergent_trials
            nonconvergents_string = f'{round(time_diff*1000)} Miliseconds Post. Estimation, {int(nonconvergent_trials_cum)} non-convergent Trials'


            # print('Aggregate Data', flush=True)
            df = pd.DataFrame(pred_data.reshape((n_obs*num_resim, 3)))

            df.columns = ['rt', 'acc', 'congruency']

            df = df[df['rt'] != -1]

            # Group by 'Group' and calculate mean and quantiles
            # Group by 'Group' and calculate aggregations for RT and ACC
            df_aggr = df.groupby("congruency").agg({
                "rt": [
                    ("mean"),
                    (lambda x: x.quantile(0.1)),
                    (lambda x: x.quantile(0.2)),
                    (lambda x: x.quantile(0.3)),
                    (lambda x: x.quantile(0.4)),
                    (lambda x: x.quantile(0.5)),
                    (lambda x: x.quantile(0.6)),
                    (lambda x: x.quantile(0.7)),
                    (lambda x: x.quantile(0.8)),
                    (lambda x: x.quantile(0.9)),
                    (lambda x: x.quantile(1))
                ],
                "acc": [
                    ( "mean"),
                ]
            }).reset_index()

            df_aggr.columns = ['congruency', 'mean_rt',
                               'qu10_rt', 'qu20_rt', 'qu30_rt',
                               'qu40_rt', 'qu50_rt', 'qu60_rt',
                               'qu70_rt', 'qu80_rt', 'qu90_rt',
                               'qu100_rt',

                               'mean_acc']

            df_aggr['mean_rt_overall'] = np.mean(df['rt'])
            df_aggr['nonconvergent_trials'] = nonconvergent_trials
            df_aggr['accuracy_overall'] = np.mean(df['acc'])

            df_aggr['n_posterior_draws'] = num_resim
            df_aggr['condition'] = index
            df_aggr['id'] = id
            df_aggr['slurm_id'] = slurm_id
            df_aggr['split'] = 'no_split'
            df_aggr['spacing'] = spacing
            df_aggr['estimation_level'] = 'individual'
            df_aggr['n_posterior_draws'] = num_resim
            df_aggr['number_parts'] = 1
            df_aggr['time_post_sampling_sec'] = time_diff

            list_pred_data_out.append(df_aggr)
            df_out = pd.concat(list_pred_data_out)
            df_out.to_csv(parent_dir + f'/data/ppc_delta_functions/ppc_delta_functions_empirical/ppc_pred_data_{index}_nosplit_spacing_individual.csv')

            p_bar.set_postfix_str(nonconvergents_string, refresh=False)
            p_bar.update(1)




