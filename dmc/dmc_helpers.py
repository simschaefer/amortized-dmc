import pandas as pd
import numpy as np
import time
import bayesflow as bf
from dmc import DMC


def load_model_specs(model_specs, network_name):

    simulator = DMC(**model_specs['simulation_settings'])

    
    if simulator.sdr_fixed == 0:

        adapter = (
            bf.adapters.Adapter()
            .drop('sd_r')
            .convert_dtype("float64", "float32")
            .sqrt("num_obs")
            .concatenate(model_specs['param_names'], into="inference_variables")
            .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
            .standardize(include="inference_variables")
            .rename("num_obs", "inference_conditions")
        )
    else:
        adapter = (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .sqrt("num_obs")
            .concatenate(model_specs['param_names'], into="inference_variables")
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
        checkpoint_name= network_name,
        inference_variables=model_specs['param_names']
    )

    return simulator, adapter, inference_net, summary_net, workflow


def format_empirical_data(data):
    
    # extract relveant variables
    data_np = data[['rt', 'accuracy', "congruency_num"]].values

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
    random_idx = np.random.choice(np.arange(0,num_resims), size = num_resims)

    # select posterior samples
    resim_samples = post_sample_data.iloc[random_idx][param_names]

    # adjust number of trials in simulator (should be equal to the number of trials in the empirical data)
    # simulator.num_obs=num_obs

    list_resim_dfs = []

    # resimulate
    for i in range(num_resims):
        resim =  simulator.experiment(A=resim_samples["A"].values[i],
                                tau=resim_samples["tau"].values[i],
                                mu_c=resim_samples["mu_c"].values[i],
                                mu_r=resim_samples["mu_r"].values[i],
                                b=resim_samples["b"].values[i],
                                num_obs=num_obs)
        
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
    
    quantile_data.rename(columns={"level_2": "quantiles"}, inplace=True)

    quantile_data_wide = quantile_data.pivot(index="quantiles", columns=congruency_name, values=rt_var)

    quantile_data_wide["delta"] = quantile_data_wide["incongruent"] - quantile_data_wide["congruent"]

    quantile_data_wide["mean_qu"] = (quantile_data_wide["incongruent"] + quantile_data_wide["congruent"])/2

    return quantile_data_wide



def subset_data(data, num_obs, idx, keys = ['rt', 'accuracy', 'conditions']):

    data = data.copy()

    for k in keys:
        # print(f'{data[k].shape}')
        data[k] = data[k][:, idx, :]
        print(f'{k}: {data[k].shape}')

    data['num_obs'] = np.array([num_obs]*1000).reshape(1000,1)

    return data

def param_labels(param_names):

    param_labels = []

    for p in param_names:

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        param_labels.append(suff + p + "$")

    if len(param_labels) <= 1:
        param_labels = param_labels[0]
        
    return param_labels