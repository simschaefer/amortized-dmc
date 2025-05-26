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

import bayesflow as bf
import time

parent_dir = os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)


from dmc import DMC

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

arguments = sys.argv[1:]
network_name = str(arguments[0])
#network_name = 'dmc_optimized_winsim_priors_sdr_fixed_200_795737'

#network_name = 'dmc_optimized_winsim_priors_sdr_fixed_150_795633'

num_resims = 100
cumulative = True


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

included_parts = np.array([
    275, 808, 810, 833, 837, 845, 916, 985, 1108, 1430, 1507, 1538, 1582, 1583, 1597, 1601,
    1614, 1638, 1657, 1663, 1761, 1768, 1813, 1821, 1824, 3286, 3292, 3487, 3580, 3625, 3754, 3910,
    3988, 4222, 4264, 5281, 5332, 5575, 5731, 5761, 5803, 5815, 6055, 6109, 6214, 6253, 6262, 6361,
    6427, 6583, 6634, 6844, 7624, 7756, 7768, 7807, 7813, 7828, 7840, 7924, 7939, 8026, 8308, 8311,
    8446, 8521, 8704, 8755, 8785, 8788, 161753, 337788
])

model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

#simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

simulator = DMC(**model_specs['simulation_settings'])
# Load checkpoints
approximator = keras.saving.load_model(parent_dir + "/bf_dmc/data/training_checkpoints/" + network_name + '.keras')

narrow_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_narrow.csv')

narrow_data = narrow_data[narrow_data['participant'].isin(included_parts)]

wide_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_wide.csv')

wide_data = wide_data[wide_data['participant'].isin(included_parts)]



empirical_data = pd.concat([narrow_data, wide_data])

samples_narrow=fit_empirical_data(narrow_data, approximator)

samples_narrow["spacing"]="narrow"

samples_wide=fit_empirical_data(wide_data, approximator)

samples_wide["spacing"]="wide"

samples_complete=pd.concat((samples_wide, samples_narrow))


parts=samples_complete["participant"].unique()



empirical_accuracies_congruent = []
empirical_accuracies_incongruent = []

resimulated_accuracies_congruent = []
resimulated_accuracies_incongruent = []


ppc_plot_folder = parent_dir + "/bf_dmc/plots/ppc/" + network_name

if not os.path.exists(ppc_plot_folder):
    os.makedirs(ppc_plot_folder)


con_color = '#132a70'
inc_color = "maroon"

# Define custom legend elements
legend_elements = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Observed'),
    Line2D([0], [0], color='black', lw=2, linestyle=':', label='Resimulated'),
    Line2D([0], [0], color=con_color, lw=2, label='Congruent'),
    Line2D([0], [0], color=inc_color, lw=2, label='Incongruent'),
]

hue_order = ["Congruent", "Incongruent"]
palette = {"Congruent": con_color, "Incongruent": inc_color}

aggr_resim_list = []


fig, axes = plt.subplots(2,2, figsize=(10,10))

#parts = [  1108,   1430,   1507,   1538,   1582,   1583,   1597,   1601]

for part in parts:
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    for spacing in (0, 1):

        if spacing == 1:
            spacing_cat = 'narrow'
        elif spacing == 0:
            spacing_cat = 'wide'
        
        # filter sample data for given participant and spacing
        part_data_samples = samples_complete[samples_complete["participant"] == part]

        part_data_samples = part_data_samples[part_data_samples['spacing'] == spacing_cat]

        # filter empirical data for given participant and narrow spacing
        part_data = empirical_data[empirical_data["participant"] == part]

        part_data = part_data[part_data["spacing_num"] == spacing]
        
        part_data["condition_label"] = part_data["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"})
        
        # compute empirical accuracy
        empirical_accuracies_congruent.append(np.mean(part_data[part_data["congruency_num"] == 0]["accuracy"]))
        empirical_accuracies_incongruent.append(np.mean(part_data[part_data["congruency_num"] == 1]["accuracy"]))
        
        
        # resimulate data
        data_resimulated =  resim_data(part_data_samples, 
                                                  num_obs=part_data.shape[0], 
                                                  num_resims=num_resims,
                                                  simulator=simulator, 
                                                  part=part, 
                                                  param_names=model_specs['simulation_settings']['param_names'] )
        

        # resim_data(post_sample_data, num_obs, simulator, part,
        
        # exclude non-convergents
        data_resimulated = data_resimulated[data_resimulated["rt"] != -1]
        
        # recode congruency
        data_resimulated["condition_label"] = data_resimulated["conditions"].map({0.0: "Congruent", 1.0: "Incongruent"})
        
        # compute resimulated data
        resimulated_accuracies_congruent.append(np.mean(data_resimulated[data_resimulated["conditions"] == 0]["accuracy"]))
        resimulated_accuracies_incongruent.append(np.mean(data_resimulated[data_resimulated["conditions"] == 1]["accuracy"]))
        
        # plot individual fit

        sns.kdeplot(part_data, x="rt", hue="condition_label", ax=axes[spacing,0], label = "Observed", hue_order=hue_order, palette=palette, linewidth=2, cumulative=cumulative)

        for resim in range(0, num_resims):
            sns.kdeplot(data_resimulated[data_resimulated['num_resim'] == resim], x="rt", hue="condition_label", ax=axes[spacing,0], linestyle="-", label = "Predicted", hue_order=hue_order, palette=palette, alpha=0.05, cumulative=cumulative)
        
        axes[spacing, 0].set_xlim(part_data['rt'].min(), part_data['rt'].max())

        aggr_data = part_data.groupby("congruency_num").mean("accuracy")

        # compute mean accuracy empirical data
        aggr_data_resim = data_resimulated.groupby("condition_label").mean("accuracy")
        aggr_data.reset_index(inplace=True)
        
        # recode congruency empirical data
        aggr_data["condition_label"] = aggr_data["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"})
        
        # compute accuracies resimulated data
        aggr_data_resim = data_resimulated.groupby(["num_resim", "conditions"]).mean("accuracy")
        aggr_data_resim.reset_index(inplace=True)
        
        # recode congruency resimulated data
        aggr_data_resim["condition_label"] = aggr_data_resim["conditions"].map({0.0: "Congruent", 1.0: "Incongruent"})


        sns.violinplot(aggr_data_resim, x="condition_label", 
                    y="accuracy", hue="condition_label", ax=axes[spacing,1], label = "Resimulated", alpha=0.5, palette=palette)
        
        axes[spacing,1].plot(aggr_data["condition_label"], aggr_data["accuracy"], "x", color="maroon", markersize=10)
        plt.ylim(0.7, 1)
        #fig.suptitle(str(part))

        #axes[spacing,0].legend(title=None, loc="best")
        
        if spacing == 1:
            axes[spacing,0].legend_.remove()
            axes[spacing,1].set_xlabel('Congruency')

        if spacing == 0:
            axes[spacing,1].set_xlabel('')

        axes[spacing,1].set_ylabel('Accuracy')

        aggr_data_resim['spacing_num'] = spacing
        aggr_resim_list.append(aggr_data_resim)
    
    # Legend 
    axes[0, 0].legend(title="", labels=hue_order)

    # Titles For RT/ACC
    axes[0,1].set_title('Accuracy')
    axes[0,0].set_title('Reaction Times')

    # Spacing Titles
    fig.text(.9, 0.3, 'Narrow', va='center', ha='left', fontsize=14, rotation=270)
    fig.text(.9, 0.75, 'Wide', va='center', ha='left', fontsize=14, rotation=270)
    fig.get_figure()


    fig.savefig(parent_dir + '/bf_dmc/plots/ppc/' + network_name + '/'  + network_name + '_' + str(part) + '.png')
    
    
df_aggr_resim = pd.concat(aggr_resim_list)

df_aggr_resim_aggr = df_aggr_resim.groupby(["participant", "condition_label", "spacing_num"]).mean(["accuracy", 'rt']).reset_index()

aggr_empirical = empirical_data.groupby(["participant", "congruency_num", "spacing_num"]).mean(["accuracy", 'rt']).reset_index()

aggr_empirical["condition_label"] = aggr_empirical["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"})
        

aggr_empirical.columns = ['participant', 'congruency_num', 'spacing_num', 'rt_empirical', 'accuracy_empirical', 'condition_label']


merged = pd.merge(aggr_empirical, df_aggr_resim_aggr, on=['participant', 'condition_label', 'spacing_num'], how='left')

merged["spacing"] = merged["spacing_num"].map({0.0: "Wide", 1.0: "Narrow"})

plt.figure()

fig, axes = plt.subplots(2, 2)

for spacing in [0, 1]:

    spacing_data = merged[merged['spacing_num'] == spacing]

    sns.scatterplot(data=spacing_data, x='rt_empirical', y='rt', hue='condition_label', ax=axes[spacing, 0], hue_order=hue_order, palette=palette, alpha=0.8)
    # Add y = x line (slope = 1, intercept = 0)
    lims = [spacing_data['rt_empirical'].min() - 0.02, 0.6]
    axes[spacing,0].plot(lims, lims, color='black', linestyle='--', linewidth=1)

    axes[spacing,0].set_ylabel('Resimulated')

    if spacing == 1:
        axes[spacing,0].set_xlabel('Empirical')
        axes[spacing,0].legend_.remove()
    else:
        axes[spacing,0].set_title('RT')
        axes[spacing,0].set_xlabel('')
        axes[spacing,0].legend().set_title("")


    sns.scatterplot(data=spacing_data, x='accuracy_empirical', y='accuracy', ax=axes[spacing, 1], hue='condition_label', hue_order=hue_order, palette=palette, alpha=0.8)

    # Add y = x line (slope = 1, intercept = 0)
    lims = [spacing_data['accuracy_empirical'].min() - 0.02, 1]
    axes[spacing, 1].plot(lims, lims, color='black', linestyle='--', linewidth=1)
    axes[spacing,1].legend_.remove()

    axes[spacing,1].set_ylabel('')

    if spacing == 1:
        axes[spacing, 1].set_xlabel('Empirical')
    else:
        axes[spacing,1].set_title('Accuracy')
        axes[spacing,1].set_xlabel('')
        axes[spacing,1].set_ylabel('')

fig.text(.98, 0.75, 'Wide', va='center', ha='left', fontsize=14, rotation=270)
fig.text(.98, 0.3, 'Narrow', va='center', ha='left', fontsize=14, rotation=270)

fig.tight_layout()

fig.savefig(parent_dir + '/bf_dmc/plots/ppc/' + network_name + '/'  + network_name + '_mean_rt_mean_acc.png')
    