import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle

import keras
import time 

import bayesflow as bf
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


arguments = sys.argv[1:]
network_name_fixed = str(arguments[0])
host = str(arguments[1])
fixed_n_obs = int(arguments[2])
num_resims = int(arguments[6])
network_name_estimated = str(arguments[3])

if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()



print(f'parent_dir: {parent_dir}', flush=True)


from dmc import DMC




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


def resim_data(post_sample_data, num_obs, simulator, part, num_resims = num_resims, param_names = ["A", "tau", "mu_c", "mu_r", "b"]):
    
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



model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name_fixed + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs_fixed = pickle.load(file)


model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name_estimated + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs_estimated = pickle.load(file)


#simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

simulator_fixed = DMC(**model_specs_fixed['simulation_settings'])
simulator_estimated = DMC(**model_specs_estimated['simulation_settings'])

# Load checkpoints
approximator_fixed = keras.saving.load_model(parent_dir + "/bf_dmc/data/training_checkpoints/" + network_name_fixed + '.keras')
approximator_estimated = keras.saving.load_model(parent_dir + "/bf_dmc/data/training_checkpoints/" + network_name_estimated + '.keras')

ppc_plot_folder = parent_dir + "/bf_dmc/plots/ppc/" + network_name_fixed

if not os.path.exists(ppc_plot_folder):
    os.makedirs(ppc_plot_folder)


narrow_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


## fixed 

samples_narrow_fixed = fit_empirical_data(narrow_data, approximator_fixed)

samples_narrow_fixed["spacing"]="narrow"

samples_wide = fit_empirical_data(wide_data, approximator_fixed)

samples_wide["spacing"]="wide"

samples_complete_fixed=pd.concat((samples_wide, samples_narrow_fixed))


## estimated 

samples_narrow_estimated = fit_empirical_data(narrow_data, approximator_estimated)

samples_narrow_estimated["spacing"]="narrow"

samples_wide = fit_empirical_data(wide_data, approximator_estimated)

samples_wide["spacing"]="wide"

samples_complete_estimated=pd.concat((samples_wide, samples_narrow_estimated))


parts=samples_complete_estimated["participant"].unique()


fig, axes = plt.subplots(6, 10, figsize=(20,12))

axes = axes.flatten()

for part, ax in zip(parts, axes):
    
    # filter sample data for given participant and narrow spacing
    part_data_samples_fixed = samples_complete_fixed[samples_complete_fixed["participant"]==part]

    part_data_samples_fixed = part_data_samples_fixed[part_data_samples_fixed["spacing"] == "narrow"]

    # filter empirical data for given participant and narrow spacing
    part_data = empirical_data[empirical_data["participant"] == part]

    part_data = part_data[part_data["spacing_num"] == 1]

    part_data["condition_label"] = part_data["congruency_num"].map({0.0: "congruent", 1.0: "incongruent"})

    # resimulate data
    data_resimulated_fixed = resim_data(part_data_samples_fixed, num_obs=part_data.shape[0], simulator=simulator_fixed, part=part, param_names=model_specs_fixed['simulation_settings']['param_names'] )
    
    # exclude non-convergents
    data_resimulated_fixed = data_resimulated_fixed[data_resimulated_fixed["rt"] != -1]

    # recode congruency
    data_resimulated_fixed["condition_label"] = data_resimulated_fixed["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

    quantiles = np.arange(0,1, 0.1)

    quantile_data_wide_resim_fixed = delta_functions(data_resimulated_fixed, quantiles = np.arange(0, 1, 0.1))

    quantile_data_wide_empirical_fixed = delta_functions(part_data, quantiles = np.arange(0, 1,0.1))

    ax.plot(quantile_data_wide_resim_fixed["mean_qu"] ,quantile_data_wide_resim_fixed["delta"] ,"-", color='#132a70', label = '$sd_r$ fixed')




        # filter sample data for given participant and narrow spacing
    part_data_samples_estimated = samples_complete_estimated[samples_complete_estimated["participant"]==part]

    part_data_samples_estimated = part_data_samples_estimated[part_data_samples_estimated["spacing"] == "narrow"]

    # filter empirical data for given participant and narrow spacing
    part_data = empirical_data[empirical_data["participant"] == part]

    part_data = part_data[part_data["spacing_num"] == 1]

    part_data["condition_label"] = part_data["congruency_num"].map({0.0: "congruent", 1.0: "incongruent"})

    # resimulate data
    data_resimulated_estimated = resim_data(part_data_samples_estimated, num_obs=part_data.shape[0], simulator=simulator_estimated, part=part, param_names=model_specs_estimated['simulation_settings']['param_names'] )
    
    # exclude non-convergents
    data_resimulated_estimated = data_resimulated_estimated[data_resimulated_estimated["rt"] != -1]

    # recode congruency
    data_resimulated_estimated["condition_label"] = data_resimulated_estimated["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

    quantiles = np.arange(0,1, 0.1)

    quantile_data_wide_resim_estimated = delta_functions(data_resimulated_estimated)

    quantile_data_wide_empirical = delta_functions(part_data, quantiles = np.arange(0, 1,0.1))

    ax.plot(quantile_data_wide_resim_estimated["mean_qu"] ,quantile_data_wide_resim_estimated["delta"] ,color = 'maroon', label = '$sd_r$ estimated')

    ax.plot(quantile_data_wide_empirical["mean_qu"] ,quantile_data_wide_empirical["delta"],"o",  color='black')

fig.tight_layout()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Mean RT')
fig.savefig(ppc_plot_folder + '/'  + network_name_estimated + '_delta_functions.png')
fig.savefig(ppc_plot_folder + '/'  + network_name_fixed + '_delta_functions.png')
    