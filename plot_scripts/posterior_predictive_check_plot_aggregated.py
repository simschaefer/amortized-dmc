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

if 'executed_from_bash' in arguments:
    network_name_fixed = str(arguments[0])
    host = str(arguments[1])
    fixed_n_obs = int(arguments[2])
    num_resims = int(arguments[6])
    network_name_estimated = str(arguments[3])

else:
    network_name_fixed = 'updated_priors_sdr_fixed'
    network_name_estimated = 'updated_priors_sdr_estimated'
    fixed_n_obs = 300
    num_resims = 100
    host = 'local'


parent_dir = os.getcwd()

print(f'parent_dir: {parent_dir}', flush=True)


from dmc import DMC, dmc_helpers


data_path = parent_dir + '/data_complete/ppc_data/'

if not os.path.exists(data_path):
      os.makedirs(data_path)

train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214])

df_complete = pd.read_csv(data_path + '/ppc_data_raw.csv')

narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])

empirical_data["condition_label"] = empirical_data["congruency_num"].map({0.0: "congruent", 1.0: "incongruent"})

empirical_data["spacing"] = empirical_data["spacing_num"].map({0.0: "wide", 1.0: "narrow"})

df_complete = df_complete[df_complete['rt'] != -1]


id_name = 'participant'

congruency_name = 'condition_label'

alpha = 0.0

n_delta_bins = 10

n_rt_bins = 5

for model in ['updated_priors_sdr_fixed','updated_priors_sdr_estimated']:

    for spacing in ['narrow', 'wide']:

        for data_split in ['train', 'test']:

            # RESIMULATED DATA
        
            data_model = df_complete[df_complete['model'] == model]

            data_model = data_model[data_model['spacing'] == spacing]


            if data_split == 'train':
                data_model = data_model[data_model['participant'].isin(train_idx)]
            else:
                data_model = data_model[~data_model['participant'].isin(train_idx)]

            delta_data = (
                data_model[data_model['accuracy'] == 1]
                .groupby([id_name, congruency_name])['rt']
                .quantile(np.arange(0.1, 1, 0.1))
                .reset_index()
                .rename(columns={'level_2': 'quantile'})
                .pivot(index=['participant', "quantile"], columns=[congruency_name], values='rt')
                .reset_index()
                .assign(delta=lambda df: df['incongruent'] - df['congruent'])
                .assign(mean_qu=lambda df: (df['incongruent'] + df['congruent'])/2)
            )

            data_model['rt_bin'] = pd.qcut(data_model['rt'], q=n_rt_bins, labels=False)

            caf_data = (
                data_model
                .groupby([id_name, congruency_name, 'rt_bin'])['accuracy']
                .mean()
                .reset_index()
                .rename(columns={'level_2': 'quantile'})
                .reset_index()
            )

            df_long = pd.melt(
                delta_data,
                id_vars=['participant', 'quantile'],
                value_vars=['congruent', 'incongruent'],
                var_name='condition',
                value_name='rt'
            )

            mean_data = df_long.groupby(['quantile', 'condition'])['rt'].mean().reset_index()


            # EMPIRICAL DATA

            data_emp = empirical_data[empirical_data['spacing'] == spacing]

            if data_split == 'train':
                data_emp = data_emp[data_emp['participant'].isin(train_idx)]
            else:
                data_emp = data_emp[~data_emp['participant'].isin(train_idx)]

            delta_data_emp = (
                data_emp[data_emp['accuracy'] == 1]
                .groupby(['participant', 'condition_label'])['rt']
                .quantile(np.arange(0.1, 1, 0.1))
                .reset_index()
                .rename(columns={'level_2': 'quantile'})
                .pivot(index=['participant', "quantile"], columns=['condition_label'], values='rt')
                .reset_index()
                .assign(delta=lambda df: df['incongruent'] - df['congruent'])
                .assign(mean_qu=lambda df: (df['incongruent'] + df['congruent'])/2)
            )

            data_emp['rt_bin'] = pd.qcut(data_emp['rt'], q=n_rt_bins, labels=False)

            caf_data_emp = (
                data_emp
                .groupby(['participant', 'condition_label', 'rt_bin'])['accuracy']
                .mean()
                .reset_index()
                .rename(columns={'level_2': 'quantile'})
                .reset_index()
            )

            df_long_emp = pd.melt(
                delta_data_emp,
                id_vars=['participant', 'quantile'],
                value_vars=['congruent', 'incongruent'],
                var_name='condition',
                value_name='rt'
            )

            mean_data_emp = df_long_emp.groupby(['quantile', 'condition'])['rt'].mean().reset_index()

            # plot RESIMULATED DATA

            hue_order = ["congruent", "incongruent"]
            palette = {"congruent": '#132a70', "incongruent": '#FF6361'}

            fig, axes = plt.subplots(1,3, figsize=(12,3))

            #sns.lineplot(df_long, linewidth=0.5, x='rt', y='quantile', hue='condition', style=id_name, legend=False, ax=axes[1], alpha=alpha, hue_order=hue_order, palette=palette)
            sns.lineplot(mean_data, linewidth=0.5,x='rt', y='quantile', hue='condition', alpha=1, ax=axes[1], legend=False, hue_order=hue_order, palette=palette)

            #sns.lineplot(df_long_emp, linewidth=0.5,linestyle='--', marker="o", x='rt', y='quantile', hue='condition', style=id_name, legend=False, ax=axes[1], alpha=alpha, hue_order=hue_order, palette=palette)
            sns.lineplot(mean_data_emp, linewidth=0.5, marker="o",linestyle='--', x='rt', y='quantile',legend=False, hue='condition', alpha=1, ax=axes[1], hue_order=hue_order, palette=palette)

            axes[0].set_title('CAF')
            axes[0].set_ylabel('CAF')
            axes[0].set_xlabel('Bins')

            sns.lineplot(caf_data, linestyle='--',marker="o", linewidth=0.5, x='rt_bin', y='accuracy', hue=congruency_name, ax=axes[0],legend=False, hue_order=hue_order, palette=palette)
            axes[1].set_title('CDF')
            axes[1].set_ylabel('Cumulative Density')
            axes[1].set_xlabel('RT[s]')
            
            sns.lineplot(caf_data_emp, errorbar=None, err_style='bars',linewidth=0.5,x='rt_bin', y='accuracy', hue=congruency_name, ax=axes[0], hue_order=hue_order, palette=palette)
            axes[0].set(ylim=(0, 1))

            axes[1].set_title('CDF')

            delta_data['mean_qu_bins'] = pd.cut(delta_data["mean_qu"], bins=n_delta_bins)
            delta_bins = delta_data.groupby('mean_qu_bins')['delta'].mean().reset_index()
            delta_bins['bin_mid'] = delta_bins['mean_qu_bins'].apply(lambda x: x.mid)


            delta_bins = (
                    delta_data
                    .groupby('quantile')[['mean_qu', 'delta']]
                    .mean()
                    .reset_index()
                    .sort_values('mean_qu')
                )

            delta_data_emp['mean_qu_bins'] = pd.cut(delta_data_emp["mean_qu"], bins=n_delta_bins)
            delta_bins_emp = delta_data_emp.groupby('mean_qu_bins')['delta'].mean().reset_index()
            delta_bins_emp['bin_mid'] = delta_bins_emp['mean_qu_bins'].apply(lambda x: x.mid)

            delta_bins_emp = (
                    delta_data_emp
                    .groupby('quantile')[['mean_qu', 'delta']]
                      .agg(
                            mean_qu=('mean_qu', 'mean'),
                            delta=('delta', 'mean'),
                            sd_delta=('delta', 'std')
                            )
                    .reset_index()
                    .sort_values('mean_qu')
                )

            #sns.lineplot(delta_data,linewidth=0.5,linestyle='--',marker="o",  x='mean_qu', y='delta', hue=id_name, legend=False, ax=axes[2], alpha=0.05)
            sns.lineplot(delta_bins,linewidth=0.5,linestyle='--',marker="o",  x='mean_qu', y='delta', legend=False, ax=axes[2], color='black')
            axes[2].set_ylabel('$\Delta$')
            axes[2].set_xlabel('RT[s]')

            #sns.lineplot(delta_data_emp,linewidth=0.5,  x='mean_qu', y='delta', hue=id_name, legend=False, ax=axes[2], alpha=alpha)
            sns.lineplot(delta_bins_emp,linewidth=0.5,  x='mean_qu', y='delta', legend=False, ax=axes[2], color='black')
    
            axes[2].set_title('$\Delta$-Function')
            
            axes[2].set(ylim=(0, 0.07))
            axes[2].set(xlim=(0.35, 0.63))
            axes[0].legend(title='', loc='lower right')

            fig.tight_layout()


            fig.savefig(parent_dir + '/plots/' + model + spacing + data_split + '_model_fit_aggregated.png')

            #sns.lineplot(delta_data_emp,linewidth=0.5,  x='mean_qu', y='delta', hue=id_name, legend=False,  alpha=1)

