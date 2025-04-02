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
    index = 536
    design_grid_num = 6
    slurm_id = 'PC'
    model_title = "dmc_603303design_grid6_condition536"
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
    'load_pretrained': False,
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
    'learning_rate': 5e-4}


if myhost == 'psyml201':
    model_info['load_pretrained'] = False
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


script_dir, parent_dir, functions_dir, model_data_dir, model_dir = find_directories(mkdir = True)

figure_path = parent_dir+"/plots/" + model_title + "/" + model_title

if model_info['load_pretrained']:
    # model titles
    list_files = os.listdir(parent_dir + '/plots')
    list_files =  [path for path in list_files if 'design_grid6' in path]
    model_title = [path for path in list_files if 'condition'+ str(index) in path]
    model_title.sort()
    model_title = model_title[0]
    model_info['model_title'] = model_title


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

f = prior.plot_prior2d(n_samples=1000)

# prior means and sds for standardizing
prior_means, prior_stds = prior.estimate_means_and_stds()

# save prior plots
if model_info['save_plots']:
    fig = f.get_figure()
    fig.savefig(parent_dir+"/plots/" + model_title + "/" + model_title +"_priors.png")


## test functions
# theta = prior_fun()

# dmc.trial(theta[0],
#           theta,
#           tmax = model_info['tmax'],
#           sigma = model_info['sigma'],
#           dt = model_info['dt'],
#           sd_r_var = model_info['sd_r_var'],
#           a_var = model_info['a_var'],
#           x0_var = model_info['x0_var'],
#           a_value = model_info['a_value'],
#           X0_value = model_info['X0_value'],
#           X0_beta_shape_fixed = model_info['X0_beta_shape_fixed'])

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

# model.simulator.context_gen.non_batchable_context_fun = lambda: model_info['max_obs']


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
                                         name="dmc_summary_"+model_info['model_title'])

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
                                             name='dmc_amortizer_'+model_info['model_title'],
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

if model_info['load_pretrained']:
    n_epochs = 0
    n_iterations = 0
else:
    n_epochs = model_info['n_epochs']
    n_iterations = model_info['n_iterations']

start_time = time.time()


h = trainer.train_online(epochs=n_epochs, iterations_per_epoch=n_iterations,
                         batch_size=model_info['batch_size'],
                         #optimizer = optimizer,
                         save_checkpoint=True,
                         early_stopping=model_info['early_stopping'],
                         validation_sims=model_info['n_validation_sims'])

end_time = time.time()


model_info['iterations_completed'] = h['train_losses'].shape[0]
model_info['fitting_time_sec'] = end_time - start_time

### example posterior draws and prior draws

num_sim = 1000
num_samples = model_info['n_posterior_samples']

model.simulator.context_gen.non_batchable_context_fun = lambda: model_info['max_obs']

# simulate data sets and apply configurator
sim_data_raw = model(batch_size=num_sim)
sim_data = trainer.configurator(sim_data_raw)

fig_rt, fig_rt_total = dmc.plot_rt(sim_data_raw)

fig_acc = dmc.plot_acc(sim_data_raw)

if model_info['save_plots']:
    fig_rt.savefig(figure_path + "_RTdist_single.png")
    fig_rt_total.savefig(figure_path + "_RTdist_total.png")
    fig_acc.savefig(figure_path + "_Accuracy.png")

# prior
prior_means_emp = sim_data['parameters']
num_obs = sim_data['summary_conditions'].shape[1]

# fit model with simulated Data
post_samples = amortizer.sample(sim_data, n_samples = num_samples)

# unstandardized posterior draws
posterior_draws_not_z = post_samples * prior_stds + prior_means

# unstandardized prior draws
priors_not_z = sim_data['parameters'] * prior_stds + prior_means

# conf_data = trainer.configurator(sim_data)
n_obs = sim_data_raw["sim_data"].shape[1]

# posterior means
post_means = np.mean(post_samples, axis = 0)

#
post_means_not_z = np.mean(posterior_draws_not_z, axis =1)

# Parameter Recovery standardized
# fig_recovery = bf.diagnostics.plot_recovery(post_samples, sim_data['parameters'],
#                                             param_names=PARAM_NAMES,
#                                             uncertainty_agg=np.std,
#                                             color="steelblue",
#                                             point_agg=np.mean)

param_not_z = sim_data['parameters'] * prior_stds + prior_means

fig_recovery_not_z = bf.diagnostics.plot_recovery(posterior_draws_not_z, param_not_z,
                                                  param_names=PARAM_NAMES,
                                                  uncertainty_agg=np.std,
                                                  color="steelblue",
                                                  point_agg=np.mean)

corr_lst = list()

for i in range(0,post_means.shape[1]):
    corr_lst.append(np.corrcoef(post_means_not_z[:,i], priors_not_z[:,i])[0][1])

for i in range(0, len(param_names)):
    param_name = param_names[i]
    model_info['r_recovery_'+param_name] = corr_lst[i]


# Check losses

fig_loss_val = bf.diagnostics.plot_losses(h["train_losses"], h["val_losses"])
# fig_loss = bf.diagnostics.plot_losses(h["train_losses"])

# Prior Predictive Check
# fig_latent = trainer.diagnose_latent2d(param_names=PARAM_NAMES)

### sbc calculation

model.simulator.context_gen.non_batchable_context_fun = sim_non_batchable_fun

num_samples_sbc = 100
bs_sbc = 2000
repetitions_sbc = 5


for rep in range(0,repetitions_sbc):
    sim_data_raw = model(batch_size=bs_sbc)
    sim_data = trainer.configurator(sim_data_raw)

    prior_samples = sim_data['parameters']
    post_samples = amortizer.sample(sim_data, n_samples = num_samples_sbc)

    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]

    rmse_sbc_lst = []

    diffs_arr = np.ones(ranks.shape)

    for j in range(len(param_names)):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])
        yy -= xx

        diffs_arr[:,j] = yy

    df = pd.DataFrame(diffs_arr)
    df.columns = param_names
    df['model_title'] = model_title
    df['post_samples_sbc'] = num_samples_sbc
    df['batch_size_sbc'] = bs_sbc
    df['repetition_sbc'] = rep

    df.to_csv(parent_dir+'/data/sbc_data/sbc_data_'+model_title+'_'+str(rep)+'.csv')

# sbc histograms
fig_sbc_trainer = trainer.diagnose_sbc_histograms(param_names=PARAM_NAMES)

fig_sbc_hist = bf.diagnostics.plot_sbc_histograms(post_samples, sim_data['parameters'],num_bins = 10,param_names=PARAM_NAMES)

# sbc ecdf
fig_sbc_ecdf = bf.diagnostics.plot_sbc_ecdf(post_samples, sim_data['parameters'], stacked=False, difference=True, legend_fontsize=10, param_names=PARAM_NAMES)

# plot posterior contraction

fig_contract = bf.diagnostics.plot_z_score_contraction(post_samples, sim_data["parameters"], param_names=PARAM_NAMES)

# plot resimulation
num_sim = 8
num_resim = 50

# Simulate and configure data
sim_data_ppc = model(batch_size=num_sim)
conf_data = trainer.configurator(sim_data_ppc)
n_obs = sim_data_ppc["sim_data"].shape[1]

# Fit model -> draw 1000 posterior samples per data set
post_samples = amortizer.sample(conf_data, n_samples=num_samples)
# Unstandardize posteriors draws into original scale
post_samples_not_z = post_samples * prior_stds + prior_means

# Generate random index for posterior parameter set selection
index_set = np.random.choice(np.arange(num_samples), size=num_resim)

# Get context of simulated data sets
context = sim_data_ppc["sim_batchable_context"]

# Re-simulate
pred_data = np.zeros((num_sim, num_resim, n_obs, 2))

for sim in range(num_sim):
    for i, idx in enumerate(index_set):
        pred_data[sim, i, :, :] = experiment_fun(post_samples_not_z[sim, idx, :], context[sim], n_obs)
        #print(f'Simulation {sim}, index {i}')

# plot resimulations

fig_ppc, axarr = plt.subplots(2, 4, figsize=(18, 8))
for i, ax in enumerate(axarr.flat):
    sns.kdeplot(
        conf_data["summary_conditions"][i, :, 0], ax=ax, fill=True, color="black", alpha=0.3, label="Simulated data"
    )
    sns.kdeplot(pred_data[i, :, :, 0].flatten(), ax=ax, fill=True, color="maroon", alpha=0.3, label="Predicted data")
    #ax.set_xlim((0, 1.5))
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(f"Simulated data set #{i+1}", fontsize=18)
    # Set legend to first plot
    if i == 0:
        ax.legend()

    # Set x label to bottom row
    if i > (num_sim // 2) - 1:
        ax.set_xlabel("Response time [ms]", fontsize=16)
    sns.despine()
fig_ppc.tight_layout()

if model_info['save_plots']:
    #fig_recovery.savefig(figure_path +"_recovery.png")
    fig_recovery_not_z.savefig(figure_path +"_recovery_not_z.png")
    fig_loss_val.savefig(figure_path + "_losses_validation.png")
    # fig_latent.savefig(figure_path + "_diagnose_latent2d.png")
    fig_sbc_trainer.savefig(figure_path + "_diagnose_sbc_trainer.png")
    fig_sbc_hist.savefig(figure_path + "_diagnose_sbc.png")
    fig_sbc_ecdf.savefig(figure_path + "_diagnose_sbc_ecdf.png")
    fig_contract.savefig(figure_path + "_z_score_contraction.png")
    fig_ppc.savefig(figure_path + "_ppc.png")

# # define index of posterior & prior draws
# id = 0
#
# fig_posterior = bf.diagnostics.plot_posterior_2d(post_samples_not_z[id], prior=model.prior, prior_draws=priors_not_z)
# #plt.axvline(priors_not_z[0])
# n_params = len(priors_not_z[id])  # Number of parameters
# vertical_lines = priors_not_z[id]  # Replace with your actual values
#
# # Get the axes from the figure
# axes = fig_posterior.get_axes()
#
# # Add vertical lines to the diagonal
# for i in range(n_params):
#     ax = axes[i * (n_params + 1)]
#     ax.axvline(vertical_lines[i], color='red', linestyle='-')
#
# plt.show()
#
# if model_info['save_plots']:
#     fig = f.get_figure()
#     fig.savefig(figure_path +"_prior_posterior.png")


#%%
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

    print(f"trial_num: {trial_nums[j]}")

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


model.simulator.context_gen.non_batchable_context_fun = sim_non_batchable_fun

if model_info['compute_mmd']:
    for i, data_set in enumerate(['acdc_clean_data_flanker.csv']):
        #save_plots = True

        print('Compute Summary Dimensions for '+data_set)
        ## read data
        data = dmc.read_data(data_set, model_data_dir, rescale_rt = False)

        # prior_fun_simple = partial(prior, context = prior_batchable_fun())

        ## compute summary statistics and data space
        sumstat, sumstat_obs, sumstat_list_obs, sumstat_list_sim = dmc.summary_space(data,
                                                                                     trainer,
                                                                                     model,
                                                                                     sim_data = True)

        # if model_info['plot_mms']:
        #     all_sumstats = pd.concat([sumstat, sumstat_obs])
        #
        #     if all_sumstats.shape[1] > 33:
        #         plt_data = all_sumstats.loc[:,list(range(0, 8)) + ['data_type']]
        #     else:
        #         plt_data = all_sumstats
        #
        #     pairplot = sns.pairplot(plt_data, hue = 'data_type', plot_kws={'alpha':0.1})
        #
        #     if model_info['save_plots']:
        #         pairplot.savefig(figure_path +  "_" + data_set + "_summarydims.png")
        #
        #     print('Pairplot computation completed')

        sumstat_np = np.delete(np.array(sumstat),-1,axis=1)
        sumstat_obs_np = np.delete(np.array(sumstat_obs),-1,axis=1)

        sumstat_np_obs = np.array(sumstat_list_obs).reshape(len(sumstat_list_obs),sumstat_list_obs[0].shape[1])
        sumstat_np_sim = np.array(sumstat_list_sim).reshape(len(sumstat_list_sim),sumstat_list_sim[0].shape[1])

        model_info['mmd_score_' + data_set] = utils.maximum_mean_discrepancy(sumstat_np_obs, sumstat_np_sim).numpy()


#%%
# Reliability even trials


obs_data = pd.read_csv(parent_dir + f'/data/model_data/experiment_data_downsampled_spacing_even.csv')

obs_data = obs_data[obs_data['spacing_num'] == 1]

parts = pd.unique(obs_data['participant'])

obs_data['interaction'] = obs_data['congruency_num'].astype('int').astype('str') + obs_data['spacing_num'].astype('int').astype('str')

for id in parts:

    part_data = obs_data[obs_data['participant'] == id]

    dir_cond = np.sqrt(part_data.shape[0])

    part_data_config = {'summary_conditions': part_data.values[np.newaxis,:,1:4].astype('float64'),
                        'direct_conditions': np.sqrt(obs_data.shape[0]).reshape(1,1)}

    posteriors_experiment = amortizer.sample(part_data_config, n_samples=1000)

    # fig_posterior = bf.diagnostics.plot_posterior_2d(posteriors_experiment, param_names=PARAM_NAMES)

    posteriors_experiment_not_z = posteriors_experiment * prior_stds + prior_means

    # fig_posterior_not_z = bf.diagnostics.plot_posterior_2d(posteriors_experiment_not_z,prior=model.prior, param_names=PARAM_NAMES)


    # fig_posterior_not_z.savefig(figure_path +"_posteriors_part275.png")



    post_data = pd.DataFrame(posteriors_experiment_not_z)

    post_data['id'] = id
    post_data['condition'] = index
    post_data['slurm_id'] = slurm_id
    post_data['split'] = 'even'
    post_data['spacing'] = 'narrow'
    post_data.to_csv(parent_dir + f'/data/posterior_experiment/posterior_data_experiment_{index}_{id}_even_narrow.csv')

### individual posteriors odd trials

obs_data = pd.read_csv(parent_dir + f'/data/model_data/experiment_data_downsampled_spacing_odd.csv')

obs_data = obs_data[obs_data['spacing_num'] == 1]

obs_data['interaction'] = obs_data['congruency_num'].astype('int').astype('str') + obs_data['spacing_num'].astype('int').astype('str')

for id in parts:

    part_data = obs_data[obs_data['participant'] == id]

    dir_cond = np.sqrt(part_data.shape[0])

    part_data_config = {'summary_conditions': part_data.values[np.newaxis,:,1:4].astype('float64'),
                        'direct_conditions': np.sqrt(obs_data.shape[0]).reshape(1,1)}

    posteriors_experiment = amortizer.sample(part_data_config, n_samples=1000)

    # fig_posterior = bf.diagnostics.plot_posterior_2d(posteriors_experiment, param_names=PARAM_NAMES)

    posteriors_experiment_not_z = posteriors_experiment * prior_stds + prior_means

    # fig_posterior_not_z = bf.diagnostics.plot_posterior_2d(posteriors_experiment_not_z,prior=model.prior, param_names=PARAM_NAMES)


    # fig_posterior_not_z.savefig(figure_path +"_posteriors_part275.png")



    post_data = pd.DataFrame(posteriors_experiment_not_z)

    post_data['id'] = id
    post_data['condition'] = index
    post_data['slurm_id'] = slurm_id
    post_data['split'] = 'odd'
    post_data['spacing'] = 'narrow'

    post_data.to_csv(parent_dir + f'/data/posterior_experiment/posterior_data_experiment_{index}_{id}_odd_narrow.csv')




obs_data = pd.read_csv(parent_dir + f'/data/model_data/experiment_data_downsampled_spacing_even.csv')

obs_data = obs_data[obs_data['spacing_num'] == 0]

parts = pd.unique(obs_data['participant'])

obs_data['interaction'] = obs_data['congruency_num'].astype('int').astype('str') + obs_data['spacing_num'].astype('int').astype('str')

for id in parts:

    part_data = obs_data[obs_data['participant'] == id]

    dir_cond = np.sqrt(part_data.shape[0])

    part_data_config = {'summary_conditions': part_data.values[np.newaxis,:,1:4].astype('float64'),
                        'direct_conditions': np.sqrt(obs_data.shape[0]).reshape(1,1)}

    posteriors_experiment = amortizer.sample(part_data_config, n_samples=1000)

    # fig_posterior = bf.diagnostics.plot_posterior_2d(posteriors_experiment, param_names=PARAM_NAMES)

    posteriors_experiment_not_z = posteriors_experiment * prior_stds + prior_means

    # fig_posterior_not_z = bf.diagnostics.plot_posterior_2d(posteriors_experiment_not_z,prior=model.prior, param_names=PARAM_NAMES)


    # fig_posterior_not_z.savefig(figure_path +"_posteriors_part275.png")



    post_data = pd.DataFrame(posteriors_experiment_not_z)

    post_data['id'] = id
    post_data['condition'] = index
    post_data['slurm_id'] = slurm_id
    post_data['split'] = 'even'
    post_data['spacing'] = 'wide'

    post_data.to_csv(parent_dir + f'/data/posterior_experiment/posterior_data_experiment_{index}_{id}_even_wide.csv')

### individual posteriors odd trials

obs_data = pd.read_csv(parent_dir + f'/data/model_data/experiment_data_downsampled_spacing_odd.csv')

obs_data = obs_data[obs_data['spacing_num'] == 0]

obs_data['interaction'] = obs_data['congruency_num'].astype('int').astype('str') + obs_data['spacing_num'].astype('int').astype('str')

for id in parts:

    part_data = obs_data[obs_data['participant'] == id]

    dir_cond = np.sqrt(part_data.shape[0])

    part_data_config = {'summary_conditions': part_data.values[np.newaxis,:,1:4].astype('float64'),
                        'direct_conditions': np.sqrt(obs_data.shape[0]).reshape(1,1)}

    posteriors_experiment = amortizer.sample(part_data_config, n_samples=1000)

    # fig_posterior = bf.diagnostics.plot_posterior_2d(posteriors_experiment, param_names=PARAM_NAMES)

    posteriors_experiment_not_z = posteriors_experiment * prior_stds + prior_means

    # fig_posterior_not_z = bf.diagnostics.plot_posterior_2d(posteriors_experiment_not_z,prior=model.prior, param_names=PARAM_NAMES)


    # fig_posterior_not_z.savefig(figure_path +"_posteriors_part275.png")



    post_data = pd.DataFrame(posteriors_experiment_not_z)

    post_data['id'] = id
    post_data['condition'] = index
    post_data['slurm_id'] = slurm_id
    post_data['split'] = 'odd'
    post_data['spacing'] = 'wide'

    post_data.to_csv(parent_dir + f'/data/posterior_experiment/posterior_data_experiment_{index}_{id}_odd_wide.csv')

#%%
# model.simulator.context_gen.non_batchable_context_fun = sim_non_batchable_fun

model_info['end_time'] = str(datetime.now())

# Convert to a dictionary with keys as index and values as data
if model_info['write_log']:
    model_info_dict = {index_value: data_value for index_value, data_value in enumerate(model_info)}

    parameter_pd = pd.DataFrame([model_info])

    # import log file
    log_file = pd.read_csv(parent_dir+'/data/'+model_info['log_file'], index_col=0)
    # #
    # # # Appending the new row to the DataFrame
    df = pd.concat([log_file, parameter_pd], ignore_index=True)

    df.to_csv(parent_dir+'/data/'+model_info['log_file'],)
