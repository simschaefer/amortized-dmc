import os


# Import Libraries
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


class BetterTimeDistributed(tf.keras.Model):
    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    @tf.function(reduce_retracing=True)
    def call(self, x, **kwargs):
        return tf.vectorized_map(self.network, x)

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
    # index = 0
    design_grid_num = int(arguments[2])
    slurm_id = str(arguments[0])
    model_title = "dmc_"+ str(arguments[0]) + 'design_grid6_hierarchical_condition' + str(index)
    design_grid_dir = '/data/design_grid' + str(design_grid_num) + '.csv'
else:
    index = 254
    design_grid_num = 6
    slurm_id = 'PC'
    model_title = "test_hierarchical_settransformers5"
    design_grid_num = 6
    # model_title = 'dmc_418664design_grid6_condition54'


    design_grid_dir = '/data/design_grid' + str(design_grid_num) + '.csv'

model_info = {
    'model_title': model_title,
    'link_function': None, # choose 'normal cdf' to transform normals to uniform priors
    'normal_restriction': 'positive_only',
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
param_names = ['A', 'b', 'mu_c', 'mu_r', 'tau']
PARAM_NAMES = [
    r'$A_{mean}$', r'$b_{mean}$',
    r'$\mu_{cmean}$', r'$\mu_{rmean}$', r'$\tau_{mean}$',
    r'$A_{sd}$', r'$b_{sd}$',
    r'$\mu_{csd}$', r'$\mu_{rsd}$', r'$\tau_{sd}$'
]


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

def normal_lim(means, sds, restriction = 'unrestricted', lower = 0, size = 1):

    if restriction == 'unrestricted':
        out = np.random.normal(means, sds, size = size)

    elif restriction == 'positive_only':
        out = np.random.normal(means, sds, size = size)

        # Keep redrawing where the condition is not met
        mask = out < lower
        while np.any(mask):
            # Redraw only for those values that did not meet the condition
            out = np.random.normal(means, sds, size = size)
            mask = out < lower

    return out


### get sd hyperpriors

# sd_hyperpriors = pd.read_csv(parent_dir + '/data/prior_x_complexity/prior_x_complexity_hyperpiors.csv')
#
# sd_hyperpriors = sd_hyperpriors[sd_hyperpriors['parameter'] != 'sd.r']
#
# prior_pars_arr_sds = sd_hyperpriors[['sd_shape', 'sd_rate']].values
#
# prior_pars_arr_sds = prior_pars_arr_sds[:5,:]

# prior_pars_arr = np.array([[40, 12],
#                            [80, 15],
#                            [0.5, 0.05],
#                            [300, 30],
#                            [80, 15]])

prior_pars_arr = np.array([[40, 6],
                           [80, 7],
                           [0.5, 0.025],
                           [300, 15],
                           [80, 7]])

prior_pars_arr_sds = np.array([[2, 0.417],
                               [2, 0.3],
                               [2, 40 ],
                               [2, 0.15],
                               [2, 0.2]])

def hyperprior(prior_pars_arr = prior_pars_arr,
                           prior_pars_arr_sds = prior_pars_arr_sds):

    prs_means = np.random.normal(prior_pars_arr[:,0], prior_pars_arr[:,1], size = prior_pars_arr.shape[0])
    prs_means[prs_means <= 0] = 0.00001
    prs_sds = np.random.gamma(prior_pars_arr_sds[:,0], 1/prior_pars_arr_sds[:,1], size = prior_pars_arr_sds.shape[0])

    hyper_prs = np.concatenate((prs_means, prs_sds), axis = 0)

    return hyper_prs


def prior_fun(theta_global, num_groups):

    theta = theta_global.reshape(2,5).T

#    theta_local = np.ones(theta.shape)
#     theta_global = hyperprior()
#     num_groups = draw_num_groups()

    theta_sbj = np.random.normal(theta[:,0], theta[:,1],  size = (num_groups, theta.shape[0]))

    theta_sbj[theta_sbj <= 0] = 0.00001

    return theta_sbj

def draw_num_groups(min_goups = 5, max_groups = 25):

    n_groups = np.random.randint(min_goups, max_groups)

    return n_groups

# prior = bf.simulation.Prior(prior_fun = hyperprior,
#                             #context_generator = prior_context,
#                             param_names = PARAM_NAMES)

# def shared_prior():
#     return np.random.uniform(low = 0, high = 1, size = 1)

prior_fun(hyperprior(), draw_num_groups())

prior = bf.simulation.TwoLevelPrior(
    hyper_prior_fun=hyperprior,
    local_prior_fun=prior_fun,
    # shared_prior_fun=shared_prior,
    local_context_generator=bf.simulation.ContextGenerator(
        non_batchable_context_fun=draw_num_groups
    ))


test_priors = prior.draw_local_parameters(prior.draw_hyper_parameters(), 10000)

f, axes = plt.subplots(nrows=1, ncols=5)
f.set_figheight(5)
f.set_figwidth(10)
for i in range(0,test_priors.shape[1]):
    axes[i].hist(test_priors[:,i])
    axes[i].set_title(param_names[i])

plt.legend()


######## Simulator function ############

@njit
def trial_exp(A,
              mu_c,
              theta,
              tmax,
              sigma,
              dt,
              sd_r_var,
              a_var,
              x0_var,
              a_value = 2,
              X0_value = 0,
              X0_beta_shape_fixed = 3):

    # ['A', 'tau', 'mu_c', 'mu_r', 'b']
    # ['A', 'tau', 'mu_c', 'mu_r', 'b', 'a']
    # ['A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r']
    # ['A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r', 'a']

    #theta = prior_fun(context_alpha())

    # tau = theta[1]
    # A = theta[0]
    tau = theta[4]
    # mu_c = theta[2]
    mu_r = theta[3]
    b = theta[1]

    if sd_r_var == 'estimated':
        sd_r = theta[5]
        # if link_fun == 'normal cdf':
        #     mu_r = norm_cdf(mu_r, mu_r_prior_mean, mu_r_prior_sd)*90+290
        #     sd_r = norm_cdf(sd_r, sd_r_prior_mean, sd_r_prior_sd)*30+15
        t0 = np.random.normal(mu_r, sd_r)
    elif sd_r_var == 'fixed':
        t0 = mu_r
        # if link_fun == 'normal cdf':
        #     t0 = norm_cdf(t0, t0_prior_mean, t0_prior_sd)*130+270

    # a estimated or fixed
    if a_var == 'fixed':
        a = a_value


    # if link_fun == 'normal cdf':
    #     A = norm_cdf(A, A_prior_mean, A_prior_sd)*25+15
    #     tau = norm_cdf(tau, tau_prior_mean, tau_prior_sd)*100+20
    #     mu_c = norm_cdf(mu_c, mu_c_prior_mean, mu_c_prior_sd)*0.6+0.2
    #     b = norm_cdf(b, b_prior_mean, b_prior_sd)*70+90

    # trial variability
    if x0_var == 'trial':
        X0 = np.random.beta(X0_beta_shape_fixed, X0_beta_shape_fixed)*(2*b)-b
    elif x0_var == 'fixed':
        X0 = X0_value


    t = np.linspace(start=dt, stop=tmax, num=int(tmax / dt))
    mu = A * np.exp((-t / tau)) * (np.exp(1) * t / (a - 1) / tau)**(a - 1) * ((a - 1) / t - 1 / tau) + mu_c
    dX = mu * dt + sigma * np.sqrt(dt) * np.random.randn(len(t))
    X_shift = np.cumsum(dX) + X0

    # plt.plot(t, X_shift)
    # #plt.xlim([0, d])
    # plt.axhline(y=b, color='r', linestyle='-')
    # plt.axhline(y=-b, color='r', linestyle='-')

    # mu_cum = np.cumsum(mu*dt)+X0
    # plt.plot(t, mu_cum)
    # plt.plot(t, X_shift)
    # plt.axhline(y=b, color='r', linestyle='-')
    # plt.axhline(y=-b, color='r', linestyle='-')

    if np.any(X_shift >= b) or np.any(X_shift <= -b):
        d = min(t[(X_shift >= b) | (X_shift <= -b)])

        rt = (d + t0)/1000

        boundary_hit = X_shift[np.where(t == d)][0]

        if boundary_hit >= b:
            # correct response
            resp = 1
        else:
            # wrong response
            resp = 0

    else:
        rt = resp = -1
    return rt, resp


theta_local = prior_fun(hyperprior(),draw_num_groups())


# n_obs_sbj_arr = sim_non_batchable_fun()
#
# n_obs_total = n_obs_sbj_arr[0]*n_obs_sbj_arr[1]
#
# obs_per_condition = np.ceil(n_obs_total / n_conditions)
# condition = np.arange(n_conditions)
# condition = np.repeat(condition, obs_per_condition)
# np.random.shuffle(condition)
# condition = condition[:n_obs_total]
# condition = condition.reshape(n_obs_sbj_arr[1], n_obs_sbj_arr[0], 1)

def condition_matrix_hierarchical(n_obs, n_conditions):

    obs_per_condition = np.ceil(n_obs / n_conditions)
    condition = np.arange(n_conditions)
    condition = np.repeat(condition, obs_per_condition)
    np.random.shuffle(condition)
    condition = condition[:n_obs]
    return condition

sim_batchable_fun = partial(condition_matrix_hierarchical, n_conditions = model_info['n_conditions'])

def random_n_obs_hierarchical(min_obs, max_obs):

    n_obs_sbj_arr = np.random.randint(min_obs, max_obs)

    return n_obs_sbj_arr


sim_non_batchable_fun = partial(random_n_obs_hierarchical, min_obs = model_info['min_obs'], max_obs = model_info['max_obs'])

non_batchable_context = sim_non_batchable_fun()

batchable_context = sim_batchable_fun(non_batchable_context)

tmax = 1200
sigma = 4
dt = 1
sd_r_var = 'fixed'
a_var = 'fixed'
x0_var = 'fixed'
a_value = 2
X0_value = 0
X0_beta_shape_fixed = 3


# @njit
def experiment_hierarchical(theta_local,
                   batchable_context,
                   non_batchable_context,
                   tmax = 1200,
                   sigma = 4,
                   dt = 1,
                   sd_r_var = 'fixed',
                   a_var = 'fixed',
                   x0_var = 'fixed',
                   a_value = 2,
                   X0_value = 0,
                   X0_beta_shape_fixed = 3):

    # theta = theta.reshape(2,5).T

    # out_lst = []
    # theta_local, theta_shared = params

    n_obs = non_batchable_context
    n_subjects = theta_local.shape[0]

    out = np.zeros((n_subjects, n_obs, 2))

    for sbj in range(0,n_subjects):

        theta_sbj = theta_local[0,:]

        # n_obs = batchable_context.shape[0]

        # mu_c
        mu_c = theta_sbj[2]

        for obs in range(n_obs):

            c = batchable_context[obs]

            # A
            A = (c * 2 - 1)*theta_sbj[0]

            rt = -1
            while rt < 0:
                rt, resp = trial_exp(A,
                                     mu_c,
                                     theta_sbj,
                                     tmax = tmax,
                                     sigma = sigma,
                                     dt = dt,
                                     sd_r_var = sd_r_var,
                                     a_var = a_var,
                                     x0_var = x0_var,
                                     a_value = a_value,
                                     X0_value = X0_value,
                                     X0_beta_shape_fixed = X0_beta_shape_fixed)
            out[sbj, obs, 0] = rt
            out[sbj, obs, 1] = resp

    return out

theta_local = prior_fun(hyperprior(), draw_num_groups())

non_batchable_context = sim_non_batchable_fun()

batchable_context = sim_batchable_fun(non_batchable_context)

experiment_hierarchical(theta_local,
                        batchable_context,
                        non_batchable_context)


sim_batchable_fun = partial(condition_matrix_hierarchical, n_conditions = model_info['n_conditions'])

non_batchable_context = sim_non_batchable_fun()

batchable_context = sim_batchable_fun(non_batchable_context)

experiment_fun = partial(experiment_hierarchical,
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

# model = bf.simulation.GenerativeModel(
#     prior=prior,
#     simulator=simulator,
#     name='DMC')

model = bf.simulation.TwoLevelGenerativeModel(
    prior=prior,
    simulator=simulator,
    name="DMC_hierarchical"
)

### sanity check

sim_data = model(10)

sim_data['sim_data'].shape

### configurator

forward_dict = model(10)

prior_draws = prior(100)

local_prior_means = np.mean(prior_draws['local_parameters'].reshape((prior_draws['local_parameters'].shape[0]*prior_draws['local_parameters'].shape[1],5)), axis = 0)
local_prior_stds = np.std(prior_draws['local_parameters'].reshape((prior_draws['local_parameters'].shape[0]*prior_draws['local_parameters'].shape[1],5)), axis = 0)

hyper_prior_means = np.mean(prior_draws['hyper_parameters'], axis = 0)

hyper_prior_stds = np.std(prior_draws['hyper_parameters'], axis = 0)


# local_prior_z = (prior_draws['local_parameters'] - local_prior_means)/local_prior_stds
#
# hyper_prior_z = (prior_draws['hyper_parameters'] - hyper_prior_means)/hyper_prior_stds


############### PRIOR MEANS & SDS!!!!!!!!!!###############



def configurator_hierarchical(forward_dict, local_prior_means, local_prior_stds, hyper_prior_means, hyper_prior_stds):

    #nonlocal prior_stds, prior_means
    # Prepare placeholder dict
    out_dict = {}

    # Get simulated data and context
    data = forward_dict['sim_data']

    # Scale reaction times so seconds
    #data[:, :, 0] /= 1000
    context = np.array(forward_dict["sim_batchable_context"])

    context = np.expand_dims(context, axis=1)  # Add a new axis at position 2 (10, 177 -> 10, 177, 1)
    context = np.repeat(context, data.shape[1], axis=1)[:,:,:,np.newaxis]


    sum_cond = np.concatenate([data, context], axis = 3).astype(np.float32)

    # Make inference network aware of varying numbers of trials
    # We create a vector of shape (batch_size, 1) by repeating the sqrt(num_obs)
    vec_num_obs = np.sqrt(forward_dict["sim_non_batchable_context"]).astype(np.float32) * np.ones((data.shape[0], 1))

    vec_num_groups = np.sqrt(forward_dict["prior_non_batchable_context"]).astype(np.float32) * np.ones((data.shape[0], 1))

    dir_global_conditions = np.concatenate((vec_num_obs, vec_num_groups), axis = 1).astype(np.float32)
    # combine alphas and number of trials:
    # dir_cond = np.concatenate((np.sqrt(vec_num_obs).astype(np.float32)), axis = 1)

    dir_local_conditions = vec_num_groups[:, np.newaxis, :].repeat(forward_dict["prior_non_batchable_context"], axis = 1).astype(np.float32)

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data))
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # Add to keys
    out_dict["summary_conditions"] = sum_cond[idx_keep][0,...]
    out_dict["direct_conditions"] = vec_num_obs[idx_keep][0,...]

    local_prior_z = (forward_dict.get("local_prior_draws") - local_prior_means)/local_prior_stds

    hyper_prior_z = (forward_dict.get("hyper_prior_draws") - hyper_prior_means)/hyper_prior_stds

    # shared_params = np.zeros((local_prior_z.shape[0],1))
    out_dict = dict(
        # shared_parameters=shared_params,
        local_parameters=local_prior_z.astype(np.float32),
        hyper_parameters=hyper_prior_z.astype(np.float32),
        summary_conditions=sum_cond[idx_keep][0,...].astype(np.float32),
        direct_global_conditions=dir_global_conditions,
        direct_local_conditions=dir_local_conditions,
    )

    return out_dict



config = partial(configurator_hierarchical,
                 local_prior_means = local_prior_means,
                 local_prior_stds = local_prior_stds,
                 hyper_prior_means = hyper_prior_means,
                 hyper_prior_stds = hyper_prior_stds)


# summary_conditions is the 4D batch of hierarchical data sets: (batch_size, num_groups, num_obs, num_dims)
# shared_parameters is of shape (batch_size, num_shared_params)
# hyper_parameters is of shape (batch_size, num_hyper_params)
# local_parameters is of shape (batch_size, num_groups, num_local_params)
# direct_global_conditions is of shape (batch_size, 2): these are the repeated (sqrt) number of groups and observations
# direct_local_conditions is of shape (batch_size, num_groups, 1): these are the repeated (sqrt) observations

config_data = config(model(13))

# config_data['shared_parameters'].shape
config_data['summary_conditions'].shape
config_data['hyper_parameters'].shape
config_data['local_parameters'].shape
config_data['direct_global_conditions'].shape
config_data['direct_local_conditions'].shape

# test = config(model(10))

#################  Defining Neural Approximator #################
#
summary_net = bf.summary_networks.HierarchicalNetwork(
    [
        BetterTimeDistributed(
            bf.networks.SetTransformer(
                input_dim=3,
                summary_dim=32
            )
        ),
        bf.networks.SetTransformer(
            num_inducing_points=16,
            input_dim=32,
            summary_dim=32,
        )
    ]
)

summary_data = config_data['summary_conditions']
summary_net(summary_data).shape
#
# summary_net = bf.summary_networks.HierarchicalNetwork([
#     bf.networks.DeepSet(summary_dim=16, num_equiv=1),
#     bf.networks.DeepSet(summary_dim=32, num_equiv=1)
# ])
#
# summary_data = config_data['summary_conditions']
# summary_net(summary_data).shape
#
# summary_net = bf.summary_networks.HierarchicalNetwork([
#     bf.networks.DeepSet(summary_dim=32)
# ])
#
# summary_data = config_data['summary_conditions']
#
# config_data['summary_conditions'].shape
# summary_net(summary_data).shape
# #
# summary_net = bf.summary_networks.HierarchicalNetwork([
#         bf.networks.SetTransformer(input_dim=3,
#                                    attention_settings=dict(key_dim=32,
#                                                            num_heads=4,
#                                                            dropout=0.0),
#                                    num_inducing_points=32,
#                                    use_layer_norm=False,
#                                    num_seeds=1,
#                                    dense_settings=dict(units=64,
#                                                        activation='relu'),
#                                    num_dense_fc=2,
#                                    summary_dim=32,
#                                    # name="dmc_summary1_"+model_info['model_title']
#                                    )
# ])
#
# summary_data = config_data['summary_conditions']
#
# summary_data.shape
#
# summary_net(summary_data).shape


# Switch off regularization for online learning
regularization_settings = {'dense_args': dict(kernel_regularizer=None), 'dropout':False}

# This will estimate p(local_k | data_k, global)
local_amortizer = bf.amortizers.AmortizedPosterior(
    bf.networks.InvertibleNetwork(
        num_params=5,
        coupling_design='spline',
        num_coupling_layers=12,
        coupling_settings=regularization_settings
    ))

# This will estimate p(global=[hyper, shared] | all_data)
global_amortizer = bf.amortizers.AmortizedPosterior(
    bf.networks.InvertibleNetwork(
        num_params=10, # 5 hyper means, 5 hyper sds
        coupling_design='spline',
        num_coupling_layers=12,
        coupling_settings=regularization_settings
    ))

# Together, p(local_k | data_k, global)and p(global=[hyper, shared] | all_data)
# define the joint posterior over all model parameters
amortizer = bf.amortizers.TwoLevelAmortizedPosterior(
    local_amortizer=local_amortizer,
    global_amortizer=global_amortizer,
    summary_net=summary_net
)

trainer = bf.trainers.Trainer(
    generative_model=model,
    configurator=config,
    amortizer=amortizer
)

# Model Summary
global_amortizer.summary()
local_amortizer.summary()
#

n_iterations = 1000
n_epochs = 5

start_time = time.time()


h = trainer.train_online(epochs=n_epochs, iterations_per_epoch=n_iterations,
                         batch_size=model_info['batch_size'],
                         #optimizer = optimizer,
                         save_checkpoint=True,
                         early_stopping=model_info['early_stopping'],
                         validation_sims=model_info['n_validation_sims'])

# amortizer.summary_net(config_data)

summary_data = config_data['summary_conditions']
# trainer.amortizer.summary_net(summary_data).shape

# trainer.amortizer.local_amortizer.summary_net(config_data)

end_time = time.time()


model_info['iterations_completed'] = h['train_losses'].shape[0]
model_info['fitting_time_sec'] = end_time - start_time

### example posterior draws and prior draws


fig_loss_val = bf.diagnostics.plot_losses(h["train_losses"], h["val_losses"])

## recovery LOCAL

num_sims_total = 100
# fit model with simulated Data
lst_post_local_samples = []
lst_post_global_samples = []
lst_prior_local_samples = []
lst_prior_global_samples = []

for i in range(num_sims_total):

    sim_data_raw = model(batch_size=1)
    sim_data = trainer.configurator(sim_data_raw)
    post_samples = amortizer.sample(sim_data, n_samples = 1000)
    lst_post_local_samples.append(post_samples['local_samples'])
    lst_post_global_samples.append(post_samples['global_samples'][np.newaxis,...])
    lst_prior_local_samples.append(sim_data['local_parameters'])
    lst_prior_global_samples.append(sim_data['hyper_parameters'])


prior_local_complete = np.concatenate(lst_prior_local_samples, axis = 1)[0,...]
prior_local_complete.shape

post_local_complete = np.concatenate(lst_post_local_samples, axis = 0)
post_local_complete.shape

PARAM_NAMES_LOCAL = ['$A$',
                     '$b$',
                     '$\\mu_{c}$',
                     '$\\mu_{r}$',
                     '$\\tau$']

fig_recovery_local = bf.diagnostics.plot_recovery(post_local_complete, prior_local_complete,
                                                  param_names=PARAM_NAMES_LOCAL,
                                                  uncertainty_agg=np.std,
                                                  color="steelblue",
                                                  point_agg=np.mean)

fig_recovery_local.savefig(figure_path +"_recovery_local.png")


# RECOVERY GLOBAL/HYPER


prior_global_complete = np.concatenate(lst_prior_global_samples, axis = 0)
prior_global_complete.shape

post_global_complete = np.concatenate(lst_post_global_samples, axis = 0)
post_global_complete.shape

fig_recovery_global = bf.diagnostics.plot_recovery(post_global_complete, prior_global_complete,
                                                   param_names=PARAM_NAMES,
                                                   uncertainty_agg=np.std,
                                                   color="steelblue",
                                                   point_agg=np.mean)

fig_recovery_global.savefig(figure_path +"_recovery_global.png")

# Check losses

fig_loss_val = bf.diagnostics.plot_losses(h["train_losses"], h["val_losses"])
# fig_loss = bf.diagnostics.plot_losses(h["train_losses"])

# Prior Predictive Check
# fig_latent = trainer.diagnose_latent2d(param_names=PARAM_NAMES)

### sbc calculation

# sbc histograms
# fig_sbc_trainer = trainer.diagnose_sbc_histograms(param_names=PARAM_NAMES)

fig_sbc_hist = bf.diagnostics.plot_sbc_histograms(post_local_complete, prior_local_complete,num_bins = 10,param_names=PARAM_NAMES_LOCAL)

# sbc ecdf
fig_sbc_ecdf_local = bf.diagnostics.plot_sbc_ecdf(post_local_complete, prior_local_complete, stacked=False, difference=True, legend_fontsize=10, param_names=PARAM_NAMES_LOCAL)

fig_sbc_ecdf_global = bf.diagnostics.plot_sbc_ecdf(post_global_complete, prior_global_complete, stacked=False, difference=True, legend_fontsize=10, param_names=PARAM_NAMES)

# plot posterior contraction

fig_contract_local = bf.diagnostics.plot_z_score_contraction(post_local_complete, prior_local_complete, param_names=PARAM_NAMES_LOCAL)


fig_contract_global = bf.diagnostics.plot_z_score_contraction(post_global_complete, prior_global_complete, param_names=PARAM_NAMES)


if model_info['save_plots']:
    #fig_recovery.savefig(figure_path +"_recovery.png")
    fig_recovery_global.savefig(figure_path +"_recovery_global.png")
    fig_recovery_local.savefig(figure_path +"_recovery_local.png")

    fig_loss_val.savefig(figure_path + "_losses_validation.png")
    # fig_latent.savefig(figure_path + "_diagnose_latent2d.png")

    fig_sbc_ecdf_local.savefig(figure_path + "_diagnose_sbc_ecdf_local.png")
    fig_sbc_ecdf_global.savefig(figure_path + "_diagnose_sbc_ecdf_global.png")
    fig_contract_local.savefig(figure_path + "_z_score_contraction_local.png")
    fig_contract_global.savefig(figure_path + "_z_score_contraction_global.png")

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
