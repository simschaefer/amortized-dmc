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

parent_dir = os.path.dirname(os.getcwd())

dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)


from dmc import DMC
from dmc import dmc_helpers

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

arguments = sys.argv[1:]
network_name = str(arguments[0])
host = str(arguments[1])
fixed_n_obs = int(arguments[2])
num_resims = int(arguments[6])

if host == 'local':
    parent_dir = '/home/administrator/Documents'
else:
    parent_dir = os.getcwd()


cumulative = True


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
approximator = keras.saving.load_model(parent_dir + "/bf_dmc/training_checkpoints/" + network_name + '.keras')

narrow_data = pd.read_csv(parent_dir + '/bf_dmc/empirical_data/experiment_data_narrow.csv')

narrow_data = narrow_data[narrow_data['participant'].isin(included_parts)]

wide_data = pd.read_csv(parent_dir + '/bf_dmc/empirical_data/experiment_data_wide.csv')

wide_data = wide_data[wide_data['participant'].isin(included_parts)]



empirical_data = pd.concat([narrow_data, wide_data])

samples_narrow = dmc_helpers.fit_empirical_data(narrow_data, approximator)

samples_narrow["spacing"]="narrow"

samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator)

samples_wide["spacing"]="wide"

samples_complete = pd.concat((samples_wide, samples_narrow))


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
        data_resimulated, excluded_samples =  dmc_helpers.resim_data(part_data_samples, 
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

### resim quantiles
data_quant_resim = pd.DataFrame(df_aggr_resim.groupby(["participant", "condition_label", "spacing_num"])['rt'].quantile([.25, .5, .75])).reset_index()

data_quant_resim_wide = data_quant_resim.pivot(columns='level_3', index = ['participant','condition_label',	'spacing_num']).reset_index()

data_quant_resim_wide.columns = ['{}_{}'.format(i, j) if j != '' else '{}'.format(i) for i, j in data_quant_resim_wide.columns]

data_quant_resim_wide.columns = ['participant', 'condition_label', 'spacing_num', 'rt_0.25_resim', 'rt_0.50_resim', 'rt_0.75_resim']



df_aggr_resim_aggr = df_aggr_resim.groupby(["participant", "condition_label", "spacing_num"]).mean(["accuracy", 'rt']).reset_index()


df_aggr_resim_aggr = pd.merge(df_aggr_resim_aggr, data_quant_resim_wide, on=['participant', 'condition_label', 'spacing_num'], how='left' )

df_aggr_resim_aggr.rename(columns={'rt': 'mean_rt_resim',
                                   'accuracy': 'mean_accuracy_resim'}, inplace=True)



## empirical data
empirical_data["condition_label"] = empirical_data["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"}) 


aggr_empirical = empirical_data.groupby(["participant", "condition_label", "spacing_num"]).mean(["accuracy", 'rt']).reset_index()

## empirical quantiles 
data_quant_emp = pd.DataFrame(empirical_data.groupby(["participant", "condition_label", "spacing_num"])['rt'].quantile([.25, .5, .75])).reset_index()

data_quant_emp_wide = data_quant_emp.pivot(columns='level_3', index = ['participant','condition_label',	'spacing_num']).reset_index()

data_quant_emp_wide.columns = ['{}_{}'.format(i, j) if j != '' else '{}'.format(i) for i, j in data_quant_emp_wide.columns]

data_quant_emp_wide.columns = ['participant', 'condition_label', 'spacing_num', 'rt_0.25_empirical', 'rt_0.50_empirical', 'rt_0.75_empirical']



aggr_empirical.columns = ['participant', 'condition_label', 'spacing_num', 'mean_rt_empirical', 'mean_accuracy_empirical', 'congruency_num']


aggr_empirical = pd.merge(aggr_empirical, data_quant_emp_wide, on=['participant', 'condition_label', 'spacing_num'], how='left' )

merged = pd.merge(aggr_empirical, df_aggr_resim_aggr, on=['participant', 'condition_label', 'spacing_num'], how='left')

merged["spacing"] = merged["spacing_num"].map({0.0: "Wide", 1.0: "Narrow"})

names = ['Mean RT', 'Mean Accuracy', '.25 Quantile RT', 'Median RT','.75 Quantile RT']

plt.figure()

fig, axes = plt.subplots(2, 5, figsize= (15,5))


for spacing in [0, 1]:

    spacing_data = merged[merged['spacing_num'] == spacing]

    for j, var in enumerate(['mean_rt', 'mean_accuracy', 'rt_0.25', 'rt_0.50', 'rt_0.75']):

        sns.scatterplot(data=spacing_data, x= var+'_empirical', y= var+'_resim', hue='condition_label', ax=axes[spacing, j], hue_order=hue_order, palette=palette, alpha=0.8, legend = False)
        # Add y = x line (slope = 1, intercept = 0)

        if var != 'mean_accuracy':
            lims = [spacing_data[var+'_empirical'].min() - 0.02, 0.6]
        else:
            lims = [.75, 1]

        axes[spacing,j].plot(lims, lims, color='black', linestyle='--', linewidth=1)

        if j == 0:
            axes[spacing,j].set_ylabel('Resimulated')
        else:
            axes[spacing,j].set_ylabel('')

        if spacing == 1:
            axes[spacing,j].set_xlabel('Empirical')
        else:
            axes[spacing,j].set_xlabel('')
        
        if spacing == 0:
            axes[spacing,j].set_title(names[j])

fig.text(.99, 0.75, 'Wide', va='center', ha='left', fontsize=14, rotation=270)
fig.text(.99, 0.3, 'Narrow', va='center', ha='left', fontsize=14, rotation=270)

fig.tight_layout()


fig.savefig(parent_dir + '/bf_dmc/plots/ppc/' + network_name + '/'  + network_name + '_mean_rt_mean_acc.png')
    


