import pandas as pd
import numpy as np
import time
import bayesflow as bf
from dmc import DMC
import copy
import warnings
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt


def load_model_specs(model_specs, network_name):

    simulator = DMC(**model_specs['simulation_settings'])

    
    if simulator.sdr_fixed == 0:

        adapter = (
            bf.adapters.Adapter()
            .drop('sd_r')
            .convert_dtype("float64", "float32")
            .sqrt("num_obs")
            .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
            .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
            .standardize(include="inference_variables")
            .rename("num_obs", "inference_conditions")
        )
    else:
        adapter = (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .sqrt("num_obs")
            .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
            .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
            .standardize(include="inference_variables")
            .rename("num_obs", "inference_conditions")
        )

    # Create inference net 
    inference_net = bf.networks.CouplingFlow(**model_specs['inference_network_settings'])

    # inference_net = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))

    summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        initial_learning_rate=model_specs['learning_rate'],
        inference_network=inference_net,
        summary_network=summary_net,
        checkpoint_filepath='../data/training_checkpoints',
        checkpoint_name=network_name,
        inference_variables=model_specs['simulation_settings']['param_names']
    )

    return simulator, adapter, inference_net, summary_net, workflow


def format_empirical_data(data, var_names=['rt', 'accuracy', "congruency_num"]):
    
    # extract relveant variables
    data_np = data[var_names].values

    # convert to dictionary
    inference_data = dict(rt=data_np[:,0],
                          accuracy=data_np[:,1],
                          conditions=data_np[:,2])

    # add dimensions so it fits training data
    inference_data = {k: v[np.newaxis,..., np.newaxis] for k, v in inference_data.items()}

    # adjust dimensions of num_obs
    inference_data["num_obs"] = np.array([data_np.shape[0]])[:,np.newaxis]
    
    return inference_data


def fit_empirical_data(data, approximator, id_label="participant"):

    ids=data[id_label].unique()

    list_data_samples=[]

    for i, id in enumerate(ids):
        
        part_data = data[data[id_label]==id]
        
        part_data = format_empirical_data(part_data)
        
        start_time=time.time()
        samples = approximator.sample(conditions=part_data, num_samples=1000)
        end_time=time.time()
        
        sampling_time=end_time-start_time

        samples_2d={k: v.flatten() for k, v in samples.items()}
        
        data_samples=pd.DataFrame(samples_2d)
        
        data_samples[id_label]=id
        data_samples["sampling_time"]=sampling_time
        
        list_data_samples.append(data_samples)

    data_samples_complete=pd.concat(list_data_samples)

    return data_samples_complete


def weighted_metric_sum(metrics_table, weight_recovery=1, weight_pc=1, weight_sbc=1):
    
    # recode posterior contraction
    metrics_table.iloc[1,:]=1-metrics_table.iloc[1,:]

    # compute means across parameters
    metrics_means=metrics_table.mean(axis=1)

    # decide on weights for each metric (Recovery, Posterior Contraction, SBC)
    metrics_weights=np.array([weight_recovery, weight_pc, weight_sbc])

    # compute weighted sum
    weighted_sum=np.dot(metrics_means, metrics_weights)
    
    return weighted_sum


def resim_data(post_sample_data, num_obs, simulator, part, num_resims = 50, param_names = ["A", "tau", "mu_c", "mu_r", "b"]):
    
    # generate random indices for random draws of posterior samples for resimulation
    random_idx = np.random.choice(np.arange(0,post_sample_data.shape[0]), size = num_resims)

    # convert to dict (allow differing number of samples per parameter)
    resim_samples = dict(post_sample_data)

    # exclude negative samples
    for k, dat in resim_samples.items():
        if k in param_names:
            resim_samples[k] = dat.values[dat.values >= 0]

    # adjust number of trials in simulator (should be equal to the number of trials in the empirical data)
    # simulator.num_obs=num_obs

    list_resim_dfs = []

    # resimulate
    for i in range(num_resims):

        if simulator.sdr_fixed is not None:
            resim =  simulator.experiment(A=resim_samples["A"][i],
                                    tau=resim_samples["tau"][i],
                                    mu_c=resim_samples["mu_c"][i],
                                    mu_r=resim_samples["mu_r"][i],
                                    b=resim_samples["b"][i],
                                    num_obs=num_obs)
        else:
            resim =  simulator.experiment(A=resim_samples["A"][i],
                        tau=resim_samples["tau"][i],
                        mu_c=resim_samples["mu_c"][i],
                        mu_r=resim_samples["mu_r"][i],
                        b=resim_samples["b"][i],
                        num_obs=num_obs,
                        sd_r=resim_samples['sd_r'][i])

        resim_df = pd.DataFrame(resim)
        
        resim_df["num_resim"] = i
        resim_df["participant"] = part
        
        list_resim_dfs.append(pd.DataFrame(resim_df))

    resim_complete = pd.concat(list_resim_dfs)
    
    return resim_complete

def delta_functions(data, quantiles = np.arange(0,1, 0.1), 
                  grouping_labels=["participant", "condition_label"],
                  rt_var="rt",
                  congruency_name="condition_label"):
    

    quantile_data = data.groupby(grouping_labels)[rt_var].quantile(quantiles).reset_index()
    
    if 'level_2' in quantile_data.columns:
        quantile_data.rename(columns={"level_2": "quantiles"}, inplace=True)

    if 'level_3' in quantile_data.columns:
        quantile_data.rename(columns={"level_3": "quantiles"}, inplace=True)

    quantile_data_wide = quantile_data.pivot(index="quantiles", columns=congruency_name, values=rt_var)

    quantile_data_wide["delta"] = quantile_data_wide["incongruent"] - quantile_data_wide["congruent"]

    quantile_data_wide["mean_qu"] = (quantile_data_wide["incongruent"] + quantile_data_wide["congruent"])/2

    return quantile_data_wide



def subset_data(data, idx, keys = ['rt', 'accuracy', 'conditions']):

    data = copy.deepcopy(data)

    for k in keys:
        # print(f'{data[k].shape}')
        data[k] = data[k][:, idx, :]
        print(f'{k}: {data[k].shape}')

    return data

def param_labels(param_names):

    param_labels = []

    for p in param_names:

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        param_labels.append(suff + p + "$")

    if len(param_labels) <= 1:
        param_labels = param_labels[0]
        
    return param_labels


def cohens_d_samples(samples1, samples2, param_names, num_samples=1000, sharex=True, subj_id='Subject', hdi_color='black', hdi_alpha=1, x_prop=0.05, y_prop=0.85, text_rotation=0, zero_line=True, x_lower=-1.2, x_upper=1.2):

    num_params = len(param_names)
    cohens_ds = np.ones((num_samples,num_params))

    parts = samples1[subj_id].unique()


    samples1.sort_values(by=subj_id, inplace=True)
    samples2.sort_values(by=subj_id, inplace=True)

    samples1['sample_id'] = np.tile(np.arange(0,num_samples), parts.shape[0])
    samples2['sample_id'] = np.tile(np.arange(0,num_samples), parts.shape[0])

    for j,p in enumerate(param_names):
        for i in range(0, num_samples):
            # control condition
            m1 = samples1[samples1['sample_id'] == i][p]
            #m1 = m1[~np.isnan(m1)]

            # experimental manipulation
            m2 = samples2[samples2['sample_id'] == i][p]
            #m2 = m2[~np.isnan(m2)]

            if set(samples1[samples1['sample_id'] == i][subj_id].unique()) != set(parts):
                warnings.warn(f'Participants in sub sample 1 and sample id {i} are not identical to all participants!')
            
            if set(samples2[samples2['sample_id'] == i][subj_id].unique()) != set(parts):
                warnings.warn(f'Participants in sub sample 2 and sample id {i} are not identical to all participants!')

            if m1.shape[0] != parts.shape[0] or m2.shape[0] != parts.shape[0]:
                warnings.warn(f'Mismatch in number of entries in sample id {i}')


            d = np.mean(m1) - np.mean(m2)
            mean_d = d/np.std(m1 - m2)

            cohens_ds[i,j] = mean_d

    data_d = pd.DataFrame(cohens_ds, columns = param_names)

    
    fig, axes = plt.subplots(1, len(param_names), figsize=(15,3), sharex=sharex)

    for p, ax in zip(param_names, axes):

        #sns.kdeplot(data=data_d, x=p, ax=ax, color=hdi_color, fill=True, alpha=hdi_alpha)
        ax.set_xlim(x_lower, x_upper)

        post_mean = np.mean(data_d[p])
        ax.axvline(x=post_mean, color='black', linestyle='--', linewidth=1)

        if zero_line:
            ax.axvline(x=0, color='red', linestyle='-', linewidth=1)

        #ax.set_xlim(x_lower, x_upper)
        hdi_bounds = az.hdi(data_d[p].values, hdi_prob=0.95)

        # HDI as shaded region with a different, subtle color
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=True, alpha=0.3,linewidth=0)
        ax.axvspan(ax.get_xlim()[0], hdi_bounds[0], color='white', alpha=1)  # Left of HDI
        ax.axvspan(hdi_bounds[1], ax.get_xlim()[1], color='white', alpha=1)  # Right of HDI
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=False, alpha=1,linewidth=1)


        #ax.axvline(hdi_bounds[0], color='gray', linestyle='--')
        #ax.axvline(hdi_bounds[1], color='gray', linestyle='--')

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        label = suff + p + "$"

        ax.set_title(label)
        ax.set_xlabel('')

        if p == 'A':
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('')

        ymax = ax.get_ylim()[1]
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]

        x_range = xmax-xmin

        ax.text(xmin + x_range*x_prop, ymax*y_prop, '$d = $' + str(round(post_mean, 2)), fontsize=12, color='black', rotation=0)
    
    fig.supxlabel('Standardized Mean Difference $d_i$', fontsize=14)
    fig.tight_layout()

    return data_d, fig