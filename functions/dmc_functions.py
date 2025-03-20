#%%
import numpy as np
import random
import bayesflow as bf
from bayesflow import computational_utilities as utils
from numba import njit
import numba as nb
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import seaborn as sns
import os
import math

from keras.utils import to_categorical


## check coding of congruency
def check_congruency(data, coding_congruent = 0, coding_incongruent = 1, rt = 'rt'):
    mean_congruent = data[data['congruency_num'] == coding_congruent][rt].mean()
    mean_incongruent = data[data['congruency_num'] == coding_incongruent][rt].mean()

    if mean_congruent >= mean_incongruent:
        print('RT IN CONGRUENT TRIALS > RT IN INCONGRUENT TRIALS! CHECK CODING!', flush = True)
    else:
        print(f'CHECK! Congruent: {coding_congruent}, Incongruent: {coding_incongruent}')

def configurator_obs(data):

    config_data = {'summary_conditions': part_data.values[np.newaxis,:,1:4].astype('float64'),
                   'direct_conditions': np.sqrt(part_data.shape[0]).reshape(1,1).astype('float64')}

    return config_data


RNG = np.random.default_rng(2023)

@njit
def trial(A,
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

    tau = theta[1]
    mu_c = theta[2]
    mu_r = theta[3]
    b = theta[4]

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
    if a_var == 'estimated':
        a = theta[-1]
    elif a_var == 'fixed':
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

    #mua <- A/tau * np.exp(1 - t/tau) * (1 - t/tau)

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
    tau = theta[1]
    # mu_c = theta[2]
    mu_r = theta[3]
    b = theta[4]

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

@njit
def trial_it(A, theta, tmax, sigma, dt):

    # dt = 0.001
    # sigma = 3
    # tmax = 15000
    # theta = DMC_prior()
    # A = theta[0]


    tau = theta[1]
    mu_c = theta[2]
    t0 = theta[3]
    b = theta[4]
    a = theta[5]
    X0 = theta[6]
    x = X0
    t = 0.0+dt

    # correct response
    resp = 1

    while t < tmax and x < b and x > -b:

        # t = np.linspace(start=dt, stop=tmax, num=int(tmax / dt))
        mu = A * np.exp((-t / tau)) * (np.exp(1) * t / (a - 1) / tau)**(a - 1) * ((a - 1) / t - 1 / tau) + mu_c
        dX = mu * dt + sigma * np.sqrt(dt) * np.random.randn()
        # X_shift = np.cumsum(dX) + X0
        x += dX
        t += dt

    rt = (t + t0)/1000

    if x <= b:
        # wrong response
        resp = 0

    return rt, resp

@njit
def experiment(theta,
                   batchable_context,
                   non_batchable_context,
                   tmax,
                   sigma,
                   dt,
                   sd_r_var,
                   a_var,
                   x0_var,
                   a_value,
                   X0_value,
                   X0_beta_shape_fixed,
               nonconvergent_warning = False,
               max_nonconvergent = 1000):

    n_obs = batchable_context.shape[0]
    out = np.zeros((n_obs, 2))

    for n in range(n_obs):
        if batchable_context[n] == 0:
            A = theta[0]
        else:
            A = -theta[0]

        rt = -1
        counter = 0
        while rt < 0:

            counter += 1

            rt, resp = trial(A, theta,
                             tmax = tmax,
                             sigma = sigma,
                             dt = dt,
                             sd_r_var = sd_r_var,
                             a_var = a_var,
                             x0_var = x0_var,
                             a_value = a_value,
                             X0_value = X0_value,
                             X0_beta_shape_fixed = X0_beta_shape_fixed)

            if counter > max_nonconvergent:
                if nonconvergent_warning:
                    print('WARNING: RESAMPLING DIFFUSION PROCESS DID NOT DIVERGE AFTER 100 REPETITIONS!')

                resp = -1

                break
        out[n, 0] = rt
        out[n, 1] = resp
    return out

@njit
def experiment_exp(theta,
               batchable_context,
               non_batchable_context,
               tmax,
               sigma,
               dt,
               sd_r_var,
               a_var,
               x0_var,
               a_value,
               X0_value,
               X0_beta_shape_fixed):

    n_obs = batchable_context.shape[0]
    out = np.zeros((n_obs, 2))

    for n in range(n_obs):

        s = batchable_context[n,1]

        c = batchable_context[n,0]

        # if batchable_context[n,1] == 0:
        #     A = theta[0] + theta[5]
        #     tau = theta[1] + theta[6]
        # else:
        #     tau = theta[0]
        #     A = theta[0]
        #
        # if batchable_context[n,0] == 0:
        #     A = A
        # else:
        #     A = -A

        # A_intercept = theta[0]
        # mu_c_intercept = theta[2]
        # d_A_spacing = theta[5]
        # d_mu_c_spacing = theta[6]
        # d_A_interaction = theta[7]
        # d_mu_c_interaction = theta[8]
        # d_mu_c_congruency = theta[9]

        # A
        A = (c * 2 - 1)*(theta[0] + s * theta[5] + c * s * theta[7])

        # mu_c
        mu_c = theta[2] + s * theta[6] + c * theta[9] + c * s * theta[8]

        rt = -1
        while rt < 0:
            rt, resp = trial_exp(A,
                                 mu_c,
                                 theta,
                                 tmax = tmax,
                                 sigma = sigma,
                                 dt = dt,
                                 sd_r_var = sd_r_var,
                                 a_var = a_var,
                                 x0_var = x0_var,
                                 a_value = a_value,
                                 X0_value = X0_value,
                                 X0_beta_shape_fixed = X0_beta_shape_fixed)
        out[n, 0] = rt
        out[n, 1] = resp
    return out

def condition_matrix(n_obs, n_conditions):
    obs_per_condition = np.ceil(n_obs / n_conditions)
    condition = np.arange(n_conditions)
    condition = np.repeat(condition, obs_per_condition)
    np.random.shuffle(condition)
    return condition[:n_obs]

def condition_matrix_exp(n_obs, n_conditions_congruency, n_conditions_effect):
    obs_per_condition = np.ceil(n_obs / n_conditions_congruency)

    condition_cong = np.arange(n_conditions_congruency)
    condition_cong = np.repeat(condition_cong, obs_per_condition)

    np.random.shuffle(condition_cong)

    condition_effect = np.arange(n_conditions_effect)
    condition_effect = np.repeat(condition_effect, obs_per_condition)

    np.random.shuffle(condition_effect)

    condition_matrix = np.vstack((condition_cong, condition_effect)).T

    return condition_matrix[:n_obs,:]

def beta_lim(lower, upper, shape):
    width = upper-lower
    return np.random.beta(shape,shape)*width+lower

def normal_lim(means, sds, restriction = 'unrestricted', lower = 0):

    if restriction == 'unrestricted':
        out = np.random.normal(means, sds)

    elif restriction == 'positive_only':
        out = np.random.normal(means, sds)

        # Keep redrawing where the condition is not met
        mask = out < lower
        while np.any(mask):
            # Redraw only for those values that did not meet the condition
            out[mask] = np.random.normal(means[mask], sds[mask])
            mask = out < lower

    return out

@njit
def norm_cdf(x, mean, sd):
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))

def prior(context,
          families,
          pars1,
          pars2,
          param_names,
          restriction,
          shift_a = True):

    prs = np.ones(len(families))  # Initialize the output array with ones
    alpha = context               # Use the provided context value

    mask_gamma = families == 'gamma'
    mask_normal = families == 'normal'
    a_mask = param_names == 'a'

    if np.any(mask_gamma):
        gamma_alpha = pars1[mask_gamma] * alpha - alpha + 1
        gamma_beta = 1 / (pars2[mask_gamma] * alpha)
        gamma_draws = np.random.gamma(gamma_alpha, gamma_beta)
        prs[mask_gamma] = gamma_draws

    if np.any(mask_normal):
        normal_draws = normal_lim(pars1[mask_normal], pars2[mask_normal] / np.sqrt(alpha), restriction)
        prs[mask_normal] = normal_draws

    # Shift 'a' distribution if required
    if shift_a:
        prs[a_mask] += 1

    # Validate the 'a' parameter
    if np.any(prs[a_mask] < 1):
        raise ValueError('ERROR: Generated "a" value is less than 1.')

    return prs

def prior_exp(families,
              pars1,
              pars2,
              param_names,
              restriction,
              shift_a = True):

    prs = np.ones(len(families))  # Initialize the output array with ones            # Use the provided context value

    mask_gamma = families == 'gamma'
    mask_normal = families == 'normal'
    a_mask = param_names == 'a'
    mask_restricted = restriction == 'positive_only'
    mask_unrestricted = restriction == 'unrestricted'

    prs = np.zeros(pars1.shape)

    if np.any(mask_gamma):
        gamma_alpha = pars1[mask_gamma]
        gamma_beta = 1 / (pars2[mask_gamma])
        gamma_draws = np.random.gamma(gamma_alpha, gamma_beta)
        prs[mask_gamma] = gamma_draws

    if np.any(mask_normal):
        # restricted:
        prs[mask_normal & mask_restricted] = normal_lim(pars1[mask_normal & mask_restricted], pars2[mask_normal & mask_restricted], restriction = 'positive_only')

        # unrestricted:
        prs[mask_normal & mask_unrestricted] = normal_lim(pars1[mask_normal & mask_unrestricted], pars2[mask_normal & mask_unrestricted], restriction = 'unrestricted')

    # Shift 'a' distribution if required
    if shift_a:
        prs[a_mask] += 1

    # Validate the 'a' parameter
    if np.any(prs[a_mask] < 1):
        raise ValueError('ERROR: Generated "a" value is less than 1.')

    return prs

def prior_nps(families,
          pars1,
          pars2,
          param_names,
          restriction,
          shift_a = True):

    prs = np.ones(len(families))  # Initialize the output array with ones
    #alpha = context               # Use the provided context value

    mask_gamma = families == 'gamma'
    mask_normal = families == 'normal'
    a_mask = param_names == 'a'

    if np.any(mask_gamma):
        gamma_alpha = pars1[mask_gamma]
        gamma_beta = 1 / (pars2[mask_gamma])
        gamma_draws = np.random.gamma(gamma_alpha, gamma_beta)
        prs[mask_gamma] = gamma_draws

    if np.any(mask_normal):
        normal_draws = normal_lim(pars1[mask_normal], pars2[mask_normal], restriction)
        prs[mask_normal] = normal_draws

    # Shift 'a' distribution if required
    if shift_a:
        prs[a_mask] += 1

    # Validate the 'a' parameter
    if np.any(prs[a_mask] < 1):
        raise ValueError('ERROR: Generated "a" value is less than 1.')

    return prs

def prior_nps_unrestricted(families,
              pars1,
              pars2,
              param_names,
              restriction,
              shift_a = True):

    prs = np.ones(len(families))  # Initialize the output array with ones
    #alpha = context               # Use the provided context value

    mask_gamma = families == 'gamma'
    mask_normal = families == 'normal'
    a_mask = param_names == 'a'

    if np.any(mask_gamma):
        gamma_alpha = pars1[mask_gamma]
        gamma_beta = 1 / (pars2[mask_gamma])
        gamma_draws = np.random.gamma(gamma_alpha, gamma_beta)
        prs[mask_gamma] = gamma_draws

    if np.any(mask_normal):
        normal_draws = np.random.normal(pars1[mask_normal], pars2[mask_normal])
        prs[mask_normal] = normal_draws

    # Shift 'a' distribution if required
    if shift_a:
        prs[a_mask] += 1

    # Validate the 'a' parameter
    if np.any(prs[a_mask] < 1):
        raise ValueError('ERROR: Generated "a" value is less than 1.')

    return prs

def alpha_gen(min, max, size):
    """Generates power-scaling parameters from a uniform distribution."""
    return RNG.uniform(min, max, size)

def random_n_obs(min_obs, max_obs):
    return np.random.randint(min_obs, max_obs)

def plot_rt(sim_out):

    batch_size = sim_out['sim_data'].shape[0]
    n_trials = sim_out['sim_data'].shape[1]
    plt.figure()
    # Plot RT Densities
    for n in range(batch_size):
        pushforward_plot = sns.kdeplot(sim_out['sim_data'][n,:, 0].flatten(),
                                       fill=False, color='steelblue', alpha=0.4)
        sns.despine()
        plt.title(f'{batch_size} simulations a {n_trials} observations')
        fig1 = pushforward_plot.get_figure()

    congruency = np.array(sim_out['sim_batchable_context']).flatten()
    accuracy = sim_out['sim_data'][:,:, 1].flatten()
    rt = sim_out['sim_data'][:,:, 0].flatten()

    data = np.vstack((congruency, accuracy, rt)).T
    data = pd.DataFrame(data, columns=['congruency', 'accuracy', 'rt'])

    plt.figure()

    rt_plot = sns.kdeplot(data, x = 'rt', hue = 'congruency')

    sns.despine()
    plt.title(f'{batch_size} simulations a {n_trials} observations')
    fig2 = rt_plot.get_figure()

    # fig2, axarr = plt.subplots(2, 5, figsize=(12, 4))
    # for i, ax in enumerate(axarr.flat):
    #     sns.histplot(sim_out["sim_data"][i, :, 0].flatten(), color="steelblue", alpha=0.75, ax=ax)
    #     sns.despine(ax=ax)
    #     ax.set_ylabel("")
    #     ax.set_yticks([])
    #     if i > 4:
    #         ax.set_xlabel("Simulated RTs (seconds)")
    #
    # fig2.tight_layout()

    return fig1, fig2
#%%

def plot_acc(sim_out):

    # Plot Response Frequencies
    batch_size = sim_out['sim_data'].shape[0]
    n_trials = sim_out['sim_data'].shape[1]
    # Count the occurrences of each value
    data = sim_out['sim_data'][:,:, 1].flatten()
    # Convert the data to integers
    data = data.astype(int)
    counts = np.bincount(data)
    # Calculate the percentage of each count
    total_counts = np.sum(counts)
    percentage = counts / total_counts * 100

    # Plot the bar chart
    fig, ax = plt.subplots()
    bar_plot = ax.bar([0, 1], counts[:2], color='steelblue', alpha=0.9)  # Use only the first two counts for binary data

    # Set the axis labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_xlabel('Response')
    ax.set_ylabel('Frequency')
    ax.set_title('Response Frequencies')

    # Add percentage labels on top of each bar
    for bar, perc in zip(bar_plot, percentage[:2]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{perc:.1f}%', ha='center', va='bottom')

    # Display the plot
    plt.title(f'{batch_size} simulations a {n_trials} observations')
    plt.show()

    return fig

# def create_sim_data_structure(data_set, set_conditions, prior_fun):
#
#     n_obs = data_set.shape[0]
#
#     return_data = {'prior_non_batchable_context': None,
#                    'prior_batchable_context': None,
#                    'prior_draws': prior_fun(),
#                    'sim_non_batchable_context': n_obs,
#                    'sim_batchable_context': set_conditions,
#                    'sim_data': data_set}
#
#     return return_data


def read_data(data_name,
              directory_path,
              rescale_rt = True,
              rm_margin = True):

    # read in data
    data = np.genfromtxt(directory_path+"/"+data_name, delimiter=',')

    # remove first row and column

    if rm_margin:
        data = data[1:,1:]

    # rescale RTs to seconds
    if rescale_rt:
        data[:,1] *= 1000

    return data


def split_data(data,
               index2 = 8, # condition
               index1 = 0, #participant
               rt_index = 1,
               corr_resp_index = 2,
               congruency_index = 3,
               two_levels = False,
               split_label2 = "condition",
               split_label1 = "part"):

    rt_lst = []
    condition_lst = []
    split_names = []

    # turn condition split off if there is no condition column:
    if data.shape[1] < index2-1:
        two_levels = False

    # identify unique condition values
    if two_levels:
        unique_values_cond = np.unique(data[:,index2])
    else:
        unique_values_cond = np.array([0])

    # identify unique participant IDs:
    unique_values_part = np.unique(data[:,index1])

    for value in unique_values_cond:

        if two_levels:
            # filter only current condition
            mask = data[:,index2] == value
            cond_data = data[mask][:,:]
        else:
            cond_data = data

        for part in unique_values_part:

            # select only data from current participant
            mask = cond_data[:,index1] == part
            exp_data = cond_data[mask][:,:]

            # add label to list
            if two_levels:
                split_names.append(split_label2 + str(int(value)) + "_" + split_label1 +str(int(part)))
            else:
                split_names.append(split_label1 + str(int(part)))

            # select only RT and congruency
            rt_lst.append(exp_data[:,[rt_index,corr_resp_index]])
            condition_lst.append(exp_data[:,congruency_index].reshape(exp_data.shape[0]))

    return rt_lst, condition_lst, split_names
#
# def data_space(data,
#                  trainer,
#                  model,
#                  prior_fun,
#                  parent_dir,
#                  model_title,
#                  data_name,
#                  plot_data_space = True,
#                  save_plots = True,
#                  match_n_batches = True,
#                  n_batches = 200,
#                  part_index = 0,
#                  RT_index = 1,
#                  corr_resp_index = 2,
#                  congruency_index = 3):
#
#     # number of observations
#     n_obs = data.shape[0]
#
#     # list all participant IDs
#     list_parts = list(np.unique(data[:,part_index]).astype(int))
#
#     # total number of participants
#     n_parts = len(list_parts)
#
#     # n_batches times simulated data sets from the well-specified model from training (for reference)
#     # reshape data to numpy array
#     data = data[:,[part_index, RT_index, corr_resp_index, congruency_index]]
#     data_reshaped = data[:,[RT_index,corr_resp_index]].reshape(RT_index,data[:,[RT_index,corr_resp_index]].shape[0],data[:,[RT_index,corr_resp_index]].shape[RT_index])
#     set_conditions = data[:,congruency_index].reshape(data.shape[0])
#     set_conditions =  set_conditions.reshape(1,set_conditions.shape[0])
#
#     # create sim_data-like structure
#     data_structured = create_sim_data_structure(data_reshaped, set_conditions, prior_fun)
#
#     # configure data
#     conf_data = trainer.configurator(data_structured)
#
#     # sample from conf data
#     x_obs = conf_data["summary_conditions"].reshape(data.shape)
#
#     #rescale RT to sec
#     x_obs[:,0] /= 1000
#
#     if match_n_batches:
#         n_batches = n_parts
#
#     simulations = trainer.configurator(model(n_batches))
#     x = simulations["summary_conditions"]
#
#     x_reshaped = x.reshape((x.shape[1]*x.shape[0],x.shape[2]))
#
#     if plot_data_space:
#         # Create a DataFrame
#         column_names = ['rt', 'ka', 'Accuracy', 'Congruency']
#         df = pd.DataFrame(data=x_reshaped, columns=column_names)
#         df_obs = pd.DataFrame(data=x_obs, columns=column_names)
#
#
#         congruency_order = [0, 1]
#         accuracy_order = [0, 1]
#         congruency_labels = ['congruent', ' incongruent']
#         accuracy_labels = ['incorrect', 'correct']
#
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
#
#         sns.kdeplot(
#             data=df, x="rt", hue= "Accuracy",
#             fill=True, common_norm=False, palette="crest",
#             alpha=.5, linewidth=0, ax=ax1,
#             hue_order=accuracy_order
#         )
#
#         sns.kdeplot(
#             data=df_obs, x="rt", hue= "Accuracy",
#             fill=True, common_norm=False, palette="crest",
#             alpha=.5, linewidth=0, ax=ax3,
#             hue_order=accuracy_order
#         )
#
#         sns.kdeplot(
#             data=df, x="rt", hue= "Congruency",
#             fill=True, common_norm=False, palette="crest",
#             alpha=.5, linewidth=0, ax=ax2,
#             hue_order=congruency_order
#         )
#
#         sns.kdeplot(
#             data=df_obs, x="rt", hue= "Congruency",
#             fill=True, common_norm=False, palette="crest",
#             alpha=.5, linewidth=0, ax=ax4,
#             hue_order=congruency_order
#         )
#
#         ax1.set_title("Model Data")
#         ax2.set_title("Model Data")
#         ax3.set_title("Observed Data")
#         ax4.set_title("Observed Data")
#         fig.tight_layout()
#
#
#         if save_plots:
#             fig.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_misspec_dataspace_" + data_name + ".png", bbox_inches='tight')
#
#     return x_reshaped, x_obs

def configurator_nps(forward_dict, prior_means, prior_stds):

    #nonlocal prior_stds, prior_means
    # Prepare placeholder dict
    out_dict = {}

    # Get simulated data and context
    data = forward_dict['sim_data']

    # Scale reaction times so seconds
    #data[:, :, 0] /= 1000
    context = np.array(forward_dict["sim_batchable_context"])[..., None]

    #alphas = np.array(forward_dict["prior_batchable_context"]).astype(np.float32)

    sum_cond = np.c_[data, context].astype(np.float32)

    # Make inference network aware of varying numbers of trials
    # We create a vector of shape (batch_size, 1) by repeating the sqrt(num_obs)
    vec_num_obs = np.sqrt(forward_dict["sim_non_batchable_context"]).astype(np.float32) * np.ones((data.shape[0], 1))

    # combine alphas and number of trials:
    # dir_cond = np.concatenate((np.sqrt(vec_num_obs).astype(np.float32)), axis = 1)

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # Add to keys
    out_dict["summary_conditions"] = sum_cond[idx_keep]
    out_dict["direct_conditions"] = vec_num_obs[idx_keep]


    # Get data generating parameters
    params = forward_dict["prior_draws"].astype(np.float32)

    out_dict["parameters"] = (params - prior_means) / prior_stds
    # Standardize parameters

    return out_dict

def configurator_nps_exp(forward_dict, prior_means, prior_stds):

    #nonlocal prior_stds, prior_means
    # Prepare placeholder dict
    out_dict = {}

    # Get simulated data and context
    data = forward_dict['sim_data']

    # Scale reaction times so seconds
    #data[:, :, 0] /= 1000
    context = np.array(forward_dict["sim_batchable_context"])

    #alphas = np.array(forward_dict["prior_batchable_context"]).astype(np.float32)

    sum_cond = np.c_[data, context].astype(np.float32)

    # Make inference network aware of varying numbers of trials
    # We create a vector of shape (batch_size, 1) by repeating the sqrt(num_obs)
    vec_num_obs = np.sqrt(forward_dict["sim_non_batchable_context"]).astype(np.float32) * np.ones((data.shape[0], 1))

    # combine alphas and number of trials:
    # dir_cond = np.concatenate((np.sqrt(vec_num_obs).astype(np.float32)), axis = 1)

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # Add to keys
    out_dict["summary_conditions"] = sum_cond[idx_keep]
    out_dict["direct_conditions"] = vec_num_obs[idx_keep]


    # Get data generating parameters
    params = forward_dict["prior_draws"].astype(np.float32)

    out_dict["parameters"] = (params - prior_means) / prior_stds
    # Standardize parameters

    return out_dict

def configurator(forward_dict, prior_means, prior_stds):

    #nonlocal prior_stds, prior_means
    # Prepare placeholder dict
    out_dict = {}

    # Get simulated data and context
    data = forward_dict['sim_data']

    # Scale reaction times so seconds
    #data[:, :, 0] /= 1000
    context = np.array(forward_dict["sim_batchable_context"])[..., None]

    alphas = np.array(forward_dict["prior_batchable_context"]).astype(np.float32)

    sum_cond = np.c_[data, context].astype(np.float32)

    # Make inference network aware of varying numbers of trials
    # We create a vector of shape (batch_size, 1) by repeating the sqrt(num_obs)
    vec_num_obs = forward_dict["sim_non_batchable_context"] * np.ones((data.shape[0], 1))

    # combine alphas and number of trials:
    dir_cond = np.concatenate((alphas, np.sqrt(vec_num_obs).astype(np.float32)), axis = 1)

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # Add to keys
    out_dict["summary_conditions"] = sum_cond[idx_keep]
    out_dict["direct_conditions"] = dir_cond[idx_keep]


    # Get data generating parameters
    params = forward_dict["prior_draws"].astype(np.float32)

    out_dict["parameters"] = (params - prior_means) / prior_stds
    # Standardize parameters

    return out_dict

def summary_space(data,
                  trainer,
                  model,
                  part_index = 0,
                  show_progress = False,
                  sim_data = False):

    # list all participant IDs
    list_parts = list(np.unique(data[:,part_index]).astype(int))

    # total number of participants
    n_parts = len(list_parts)

    # prepare empty list
    list_n_per_part = list()

    # number of trials per participant
    for part in list_parts:
        list_n_per_part.append(data[data[:,part_index] == part,].shape[0])

    # minimum number of trials per participant
    #min_n_per_part = min(list_n_per_part)

    # prepare list for trial data by participant
    sumstat_list_obs = list()
    sumstat_list_sim = list()


    for i, part in enumerate(list_parts):
        # extract only participant data
        part_data = data[np.newaxis, data[:,part_index] == part,1:]

        sumstat_list_obs.append(np.array(trainer.amortizer.summary_net(part_data)))

        if show_progress:
            print('Simulating data corresponding to participant ' + str(part))

        if sim_data:
            model.simulator.context_gen.non_batchable_context_fun = lambda: list_n_per_part[i]

            simulations = trainer.configurator(model(1))
            x = simulations["summary_conditions"]

            sumstat_list_sim.append(np.array(trainer.amortizer.summary_net(x)))

    summary_statistics_obs = np.array(sumstat_list_obs).reshape((n_parts,sumstat_list_obs[0].shape[1]))

    sumstat_obs = pd.DataFrame(summary_statistics_obs)
    sumstat_obs['data_type'] = "observed data"

    if sim_data:
        summary_statistics = np.array(sumstat_list_sim).reshape((n_parts,sumstat_list_sim[0].shape[1]))
        sumstat = pd.DataFrame(summary_statistics)
        sumstat['data_type'] = "simulated data"

    # n_batches times simulated data sets from the well-specified model from training (for reference)


    # all_sumstats = pd.concat([sumstat, sumstat_obs])
    # pairplot = sns.pairplot(all_sumstats, hue = 'data_type')
    # plt.subplots_adjust(top=0.2)  # Adjust the subplot parameters to make room for the title
    # pairplot.fig.suptitle("Pair Plot", fontsize = 20)
    #plt.legend(title='Hue', loc='center left')

    print('Summary Statistics computed')

    if sim_data:
        return sumstat, sumstat_obs, sumstat_list_obs, sumstat_list_sim
    else:
        return sumstat_obs, sumstat_list_obs

def compute_eta(trainer,
                model,
                amortizer,
                prior_means,
                prior_stds,
                num_sim = 1000,
                num_samples = 1000):

    sim_data = trainer.configurator(model(batch_size=num_sim))
    prior_means_emp = sim_data['parameters']
    num_obs = sim_data['summary_conditions'].shape[1]


    # fit model with simulated Data
    post_samples = amortizer.sample(sim_data, n_samples = num_samples)
    #prior_samples = sim_data['parameters'] * prior_stds + prior_means


    post_samples_means = np.mean(post_samples, axis = 1)

    abs_matrix = abs(prior_means_emp - post_samples_means)

    n_cols = abs_matrix.shape[1]

    eta_lst = list()

    def compute_range(arr):
        range_out = np.max(arr) - np.min(arr)
        return range_out

    for i in range(n_cols):
        rang = compute_range(abs_matrix[:, i])
        eta = np.sum(abs_matrix[:, i]/rang)
        eta_lst.append(eta)

    return eta_lst


def plot_diagnostics(trainer,
                     h,
                     model,
                     amortizer,
                     prior,
                     param_names,
                     save_plots,
                     parent_dir,
                     model_title,
                     prior_means,
                     prior_stds,
                     experiment_fun,
                     num_sim = 1000,
                     num_samples = 1000,
                     num_resim = 50,
                     loss_validation = False):

    # Simulate some Data to fit
    sim_data = trainer.configurator(model(batch_size=num_sim))
    prior_means_emp = sim_data['parameters']
    num_obs = sim_data['summary_conditions'].shape[1]

    # fit model with simulated Data
    post_samples = amortizer.sample(sim_data, n_samples = num_samples)
    #prior_samples = sim_data['parameters'] * prior_stds + prior_means

    post_means = np.mean(post_samples, axis = 1)

    # fit model with simulated Data
    post_samples = amortizer.sample(sim_data, n_samples = num_samples)
    # unstandardized posteriors
    post_samples_not_z = post_samples * prior_stds + prior_means

    # calculate correlations between simulated and recovered:
    # corr_lst = list()
    #
    # for i in range(0,post_means.shape[1]):
    #     corr_lst.append(np.corrcoef(post_means[:,i], prior_means_emp[:,i])[0][1])

    # Some Plotting
    # Parameter Recovery
    fig_recovery = bf.diagnostics.plot_recovery(post_samples, sim_data['parameters'],
                                                param_names=param_names,
                                                uncertainty_agg=np.std,
                                                color="steelblue",
                                                point_agg=np.mean)

    if save_plots:
        fig_recovery.savefig(parent_dir+"/plots/"+ model_title + "/" + model_title +"_recovery.png")


    param_not_z = sim_data['parameters'] * prior_stds + prior_means

    fig_recovery_not_z = bf.diagnostics.plot_recovery(post_samples_not_z, param_not_z,
                                                param_names=param_names,
                                                uncertainty_agg=None,
                                                color="steelblue")

    if save_plots:
        fig_recovery_not_z.savefig(parent_dir+"/plots/"+ model_title + "/" + model_title +"_recovery_not_z.png")


    # Check losses

    if loss_validation:
        f_val = bf.diagnostics.plot_losses(h["train_losses"], h["val_losses"])
        f = bf.diagnostics.plot_losses(h["train_losses"])
        if save_plots:
            f_val.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_losses_validation.png")
    else:
        f = bf.diagnostics.plot_losses(h)

    if save_plots:
        f.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_losses.png")

    if save_plots:
        f.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_losses.png")

    # Prior Predictive Check
    fig_latent = trainer.diagnose_latent2d(param_names=param_names)
    if save_plots:
        fig_latent.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_diagnose_latent2d.png")

    # sbc histograms
    fig_sbc = trainer.diagnose_sbc_histograms(param_names=param_names)
    if save_plots:
        fig_sbc.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_diagnose_sbc.png")

    # Create ECDF plot
    f = bf.diagnostics.plot_sbc_ecdf(post_samples, sim_data['parameters'], stacked=False, difference=True, legend_fontsize=10, param_names=param_names)

    if save_plots:
        f.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_sbc_ecdf.png")

    # z-score contraction
    f = bf.diagnostics.plot_z_score_contraction(post_samples, sim_data["parameters"], param_names=param_names)
    if save_plots:
        f.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_z_score_contraction.png")


    num_sim = 8

    # Simulate and configure data
    sim_data = model(batch_size=num_sim)
    conf_data = trainer.configurator(sim_data)
    n_obs = sim_data["sim_data"].shape[1]

    # Fit model -> draw 1000 posterior samples per data set
    post_samples = amortizer.sample(conf_data, n_samples=num_samples)
    # Unstandardize posteriors draws into original scale
    post_samples_not_z = post_samples * prior_stds + prior_means

    # Generate random index for posterior parameter set selection
    index_set = np.random.choice(np.arange(num_samples), size=num_resim)

    # Get context of simulated data sets
    context = sim_data["sim_batchable_context"]

    # Re-simulate
    pred_data = np.zeros((num_sim, num_resim, n_obs, 2))
    for sim in range(num_sim):
        for i, idx in enumerate(index_set):
            pred_data[sim, i, :, :] = experiment_fun(post_samples_not_z[sim, idx, :], context[sim], n_obs)

    # plot resimulations

    f, axarr = plt.subplots(2, 4, figsize=(18, 8))
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
    f.tight_layout()
    if save_plots:
        f.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_ppc.png")


#
# def summary_space(summary_statistics,
#                 summary_statistics_obs,
#                 data_name,
#                 save_plots,
#                 model_title,
#                 parent_dir,
#                 write_data = True,
#                 n_dims = 32,
#                 plot_space = False):
#
#     n_data_sets_visualization = 10
#     colors = cm.viridis(np.linspace(0, 1, n_data_sets_visualization))
#     indices = list(range(n_data_sets_visualization))
#
#     mmd_score = utils.maximum_mean_discrepancy(summary_statistics, summary_statistics_obs)
#
#     if write_data:
#
#         summary_stats_c = np.concatenate((summary_statistics[:,0:n_dims], summary_statistics_obs[:,0:n_dims]), axis=1)
#         mmd_column = np.full((summary_statistics[:,0:n_dims].shape[0], 1), mmd_score.numpy())
#         model_title_column = np.full((summary_statistics[:,0:n_dims].shape[0], 1), model_title)
#         data_name_column = np.full((summary_statistics[:,0:n_dims].shape[0], 1), data_name)
#
#         data_out = np.hstack((summary_stats_c, mmd_column, model_title_column, data_name_column))
#
#         path = parent_dir+"/data/summary_statistics/" + model_title
#
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         np.savetxt(path + "/" + model_title +"_sumstat_"+data_name+"_"+str(n_dims)+"dims.csv", data_out, delimiter=',',  fmt='%s')
#
#     if plot_space:
#         fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#         colors = cm.viridis(np.linspace(0.1, 0.9, 2))
#         ax.scatter(
#             summary_statistics_obs[:, 0], summary_statistics_obs[:, 1], color=colors[0], label=r"Observed data: $h_{\psi}(x_{obs})$"
#         )
#         ax.scatter(summary_statistics[:, 0], summary_statistics[:, 1], color=colors[1], label=r"Model data: $h_{\psi}(x)$")
#         ax.legend()
#         ax.grid(alpha=0.2)
#         plt.gca().set_aspect("equal")
#
#         plt.suptitle("MMD: "+str(round(mmd_score.numpy(),4)), fontsize=12, y=0.92)
#         sns.despine(ax=ax)
#
#         if save_plots:
#             fig.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_misspec_summaryspace_"+ data_name + ".png", bbox_inches='tight')


def plot_data_space(x, x_obs, parent_dir, model_title, data_name, save_plots):

    # Reshape to a 2D array
    reshaped_array = x.reshape(-1, x.shape[-1])
    reshaped_array_obs = x_obs.reshape(-1, x_obs.shape[-1])
    # Create a DataFrame
    column_names = ['rt', 'ka', 'Accuracy', 'Congruency']
    df = pd.DataFrame(data=reshaped_array, columns=column_names)
    df_obs = pd.DataFrame(data=reshaped_array_obs, columns=column_names)


    congruency_order = [0, 1]
    accuracy_order = [0, 1]
    congruency_labels = ['congruent', ' incongruent']
    accuracy_labels = ['incorrect', 'correct']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    sns.kdeplot(
        data=df, x="rt", hue= "Accuracy",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, ax=ax1,
        hue_order=accuracy_order
    )

    sns.kdeplot(
        data=df_obs, x="rt", hue= "Accuracy",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, ax=ax3,
        hue_order=accuracy_order
    )

    sns.kdeplot(
        data=df, x="rt", hue= "Congruency",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, ax=ax2,
        hue_order=congruency_order
    )

    sns.kdeplot(
        data=df_obs, x="rt", hue= "Congruency",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0, ax=ax4,
        hue_order=congruency_order
    )

    ax1.set_title("Model Data")
    ax2.set_title("Model Data")
    ax3.set_title("Observed Data")
    ax4.set_title("Observed Data")
    fig.tight_layout()


    if save_plots:
        fig.savefig(parent_dir+"/plots/" + model_title + "/" + model_title + "_misspec_dataspace_" + data_name + ".png", bbox_inches='tight')

# def mmd_hypo_test(data,
#                   trainer,
#                   model,
#                   prior_fun,
#                   save_plots,
#                   data_name,
#                   parent_dir,
#                   model_title,
#                   part_index = 0,
#                   RT_index = 1,
#                   corr_resp_index = 2,
#                   congruency_index = 3):
#
#     list_parts = list(np.unique(data[:,part_index]).astype(int))
#
#     #n_parts = len(list_parts)
#
#     list_n_per_part = list()
#
#     for part in list_parts:
#         list_n_per_part.append(data[data[:,part_index] == part,].shape[0])
#
#     min_n_per_part = min(list_n_per_part)
#
#     data_per_part_list = list()
#
#     for part in list_parts:
#         part_data = data[data[:,part_index] == part,]
#
#         n_obs = part_data.shape[0]
#
#         samples = random.sample(list(range(0,n_obs)), min_n_per_part)
#
#         part_data_sampled = part_data[samples,:]
#
#         part_data_reshaped = part_data_sampled.reshape(1,part_data_sampled.shape[0], part_data_sampled.shape[1])
#
#         # # reshape data to numpy array
#         # data_reshaped = part_data_sampled[:,[RT_index,corr_resp_index]].reshape(RT_index,part_data_sampled[:,[RT_index,corr_resp_index]].shape[0],part_data_sampled[:,[RT_index,corr_resp_index]].shape[RT_index])
#         # set_conditions = part_data_sampled[:,congruency_index].reshape(part_data_sampled.shape[0])
#         # set_conditions =  set_conditions.reshape(1,set_conditions.shape[0])
#         #
#         # # create sim_data-like structure
#         # data_structured = create_sim_data_structure(data_reshaped, set_conditions, prior_fun)
#         #
#         # # configure data
#         # conf_data = trainer.configurator(data_structured)
#         #
#         # x_obs_part = conf_data["summary_conditions"]
#
#         data_per_part_list.append(part_data_reshaped)
#
#     data_per_part = np.concatenate(data_per_part_list)
#
#     # reshape data to numpy array
#     rt_add_data = data_per_part[:,:,[RT_index,corr_resp_index]]
#     set_conditions = data_per_part[:,:,congruency_index]
#     #set_conditions =  set_conditions.reshape(1,set_conditions.shape[0])
#
#     # create sim_data-like structure
#     data_structured = create_sim_data_structure(rt_add_data, set_conditions)
#
#     reference_data = trainer.configurator(model(rt_add_data.shape[0]))
#     observed_data = trainer.configurator(data_structured)
#
#     MMD_sampling_distribution, MMD_observed = trainer.mmd_hypothesis_test(
#         observed_data, reference_data=reference_data, num_reference_simulations=1000, num_null_samples=500, bootstrap=True)
#
#     _ = bf.diagnostics.plot_mmd_hypothesis_test(MMD_sampling_distribution, MMD_observed)
#
#     if save_plots:
#         _.savefig(parent_dir+"/plots/"+ model_title + "/" + model_title +"_MMD_"+data_name+".png")
#
#     return MMD_sampling_distribution, MMD_observed
#
# def compute_posteriors(rt_lst,
#                        condition_lst,
#                        split_names,
#                        data_name,
#                        prior_fun,
#                        trainer,
#                        amortizer,
#                        prior_stds,
#                        prior_means,
#                        save_posterior_data,
#                        model_title,
#                        parent_dir,
#                        num_samples = 1000,
#                        level = "condition_part"):
#
#     posterior_lst = []
#
#     data_structured_lst = []
#
#     conf_data_lst = []
#
#     for i in range(0, len(rt_lst)):
#         rt_data = rt_lst[i].reshape(1,rt_lst[i].shape[0],rt_lst[i].shape[1])
#         condition_data = condition_lst[i].reshape(1,condition_lst[i].shape[0])
#
#         data_structured = create_sim_data_structure(rt_data, condition_data, prior_fun)
#         data_structured_lst.append(data_structured)
#
#         conf_data = trainer.configurator(data_structured)
#         conf_data_lst.append(conf_data)
#
#         # Fit model -> draw 1000 posterior samples per data set
#         post_samples = amortizer.sample(conf_data, n_samples=num_samples)
#
#         # Unstandardize posteriors draws into original scale
#         post_samples_not_z = post_samples * prior_stds + prior_means
#
#         posterior_lst.append(post_samples_not_z)
#
#         data_set_name = data_name.rstrip('.csv')
#
#         path = parent_dir+"/data/posteriors/" + model_title
#
#         model_name_col = np.full((post_samples_not_z.shape[0], 1), model_title)
#
#         data_set_col = np.full((post_samples_not_z.shape[0], 1), data_set_name)
#
#         subset_col = np.full((post_samples_not_z.shape[0], 1), split_names[i])
#
#         level_col = np.full((post_samples_not_z.shape[0], 1), level)
#
#         data_out = np.concatenate((post_samples_not_z,
#                                    model_name_col,
#                                    data_set_col,
#                                    subset_col,
#                                    level_col), axis=1)
#
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         if save_posterior_data:
#             np.savetxt(path + "/posterior_" + model_title + "_" + data_set_name + "_" + split_names[i] + "_" + level +".csv",
#                        data_out,
#                        delimiter=',',
#                        fmt='%s')