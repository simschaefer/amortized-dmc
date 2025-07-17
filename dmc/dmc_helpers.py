import pandas as pd
import numpy as np
import time
import bayesflow as bf
from dmc import DMC
import copy
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


def hdi(samples, hdi_prob=0.95):
    """
    Compute the Highest Density Interval (HDI) of a sample distribution.

    Parameters
    ----------
    samples : array-like
        1D array of posterior samples.
    hdi_prob : float
        The desired probability for the HDI (e.g., 0.95 for 95% HDI).

    Returns
    -------
    hdi_interval : tuple
        Lower and upper bounds of the HDI.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("Only 1D arrays are supported.")
    
    sorted_samples = np.sort(samples)
    n_samples = len(sorted_samples)
    interval_idx_inc = int(np.floor(hdi_prob * n_samples))
    n_intervals = n_samples - interval_idx_inc

    if n_intervals <= 0:
        raise ValueError("Not enough samples for the desired HDI probability.")

    intervals = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(intervals)

    hdi_min = sorted_samples[min_idx]
    hdi_max = sorted_samples[min_idx + interval_idx_inc]

    return hdi_min, hdi_max


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
    """
    Formats empirical behavioral data into a structured dictionary for model inference.

    This function extracts specified variables from a pandas DataFrame, converts them 
    to a NumPy-based dictionary format, and reshapes the data to align with the expected 
    input dimensions of a probabilistic model or training pipeline.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing empirical data, typically with columns representing 
        response time ('rt'), accuracy, and experimental conditions.
    
    var_names : list of str, optional
        A list of column names to extract from the DataFrame. Defaults to 
        ['rt', 'accuracy', 'congruency_num'].

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - 'rt': 3D NumPy array of response times, shape (1, N, 1)
        - 'accuracy': 3D NumPy array of accuracy values, shape (1, N, 1)
        - 'conditions': 3D NumPy array of experimental condition identifiers, shape (1, N, 1)
        - 'num_obs': 2D NumPy array with the number of observations, shape (1, 1)
        
    Notes:
    ------
    The reshaping to 3D (and 2D for 'num_obs') ensures compatibility with batch-based 
    inference or training procedures where dimensions typically follow the pattern 
    (batch, number of observations, variable).
    """
    
    # extract relevant variables
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


def fit_empirical_data(data, approximator, id_label="participant", var_names=['rt', 'accuracy', "congruency_num"]):
    """
    Samples posteriors for empirical data for each unique subject or group.

    This function iterates over unique identifiers in the input DataFrame (e.g., participants),
    formats their data appropriately, performs posterior sampling using the specified 
    approximator, and aggregates the results into a combined DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing empirical observations. Must include a column corresponding
        to `id_label` to distinguish between different units (e.g., participants).
    
    approximator : bayesflow.approximators.ContinuousApproximator
        A trained BayesFlow `ContinuousApproximator` object used to perform amortized 
        posterior inference. It must implement a `.sample(conditions, num_samples)` method,
        where `conditions` is a dictionary of formatted input data and `num_samples` 
        is the number of posterior samples to draw.

    id_label : str, optional
        The column name used to identify unique units in the data (e.g., "participant").
        Defaults to "participant".

    var_names : str, optional
        Contains a list of variable names that are used as inference variables by the adapter. 
        It should contain the variable name of the reaction times (default = 'rt'), the name of the accuracy variable
        (default = 'accuracy') as well as the name of the congruency variable (default = 'congruency_num').

    Returns:
    --------
    pandas.DataFrame
        A concatenated DataFrame containing posterior samples for all individuals.
        Includes:
        - Flattened posterior samples (one column per variable)
        - The participant/group identifier (`id_label`)
        - The sampling time for each individual (`sampling_time`)

    Notes:
    ------
    - This function assumes that the `format_empirical_data` function is available
      and correctly formats individual data into a dictionary suitable for the 
      approximator.
    - The `approximator` must support a `sample` method with arguments:
      `conditions` (dict) and `num_samples` (int).
    """

    # extract unique id labels
    ids=data[id_label].unique()

    list_data_samples=[]

    # iterate over participants
    for i, id in enumerate(ids):
        
        # select participant data
        part_data = data[data[id_label]==id]
        
        # bring it into the right format (dictionary)
        part_data = format_empirical_data(part_data, var_names=var_names)    

        # draw posterior samples with the given approximator
        start_time=time.time()
        samples = approximator.sample(conditions=part_data, num_samples=1000)
        end_time=time.time()
        
        # computing total sampling time
        sampling_time=end_time-start_time

        # reformat it back into a numpy array -> DataFrame
        samples_2d={k: v.flatten() for k, v in samples.items()}
        
        data_samples=pd.DataFrame(samples_2d)
        
        data_samples[id_label]=id
        data_samples["sampling_time"]=sampling_time
        
        list_data_samples.append(data_samples)

    # combine data frames from all participants
    data_samples_complete=pd.concat(list_data_samples)

    return data_samples_complete


def weighted_metric_sum(metrics_table, weight_recovery=1, weight_pc=1, weight_sbc=1):
    """
    Computes a weighted sum of model evaluation metrics to produce a single scalar score.

    This function takes a table of metrics (e.g., parameter recovery, posterior contraction, 
    simulation-based calibration) and computes a weighted average score that can be used 
    to compare models or configurations. The second row (posterior contraction) is transformed 
    by subtracting it from 1, assuming smaller values are better.

    Parameters:
    -----------
    metrics_table : pandas.DataFrame or numpy.ndarray
        A 2D structure where each row corresponds to a different metric and each column 
        corresponds to a parameter or evaluation dimension. The expected row order is:
        0 - Recovery
        1 - Posterior Contraction (will be inverted internally)
        2 - Simulation-Based Calibration (SBC)

    weight_recovery : float, optional
        Weight assigned to the recovery metric. Default is 1.

    weight_pc : float, optional
        Weight assigned to the posterior contraction metric. Default is 1.

    weight_sbc : float, optional
        Weight assigned to the SBC metric. Default is 1.

    Returns:
    --------
    float
        A single scalar value representing the weighted sum of the mean metrics across parameters.

    Notes:
    ------
    - Posterior contraction values are assumed to be better when smaller, so they are
      transformed using `1 - value` to reward narrower posteriors.
    - All metrics are averaged across parameters before weighting.
    - This function assumes the metrics are in the expected row order.
    """
    
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
    """
    Resimulates data based on posterior parameter samples for a given participant.

    This function takes posterior samples, filters out invalid values (e.g., negatives), and uses 
    them to generate synthetic datasets by repeatedly calling a simulator. It supports both fixed 
    and variable `sd_r` scenarios depending on the simulator configuration.

    Parameters:
    -----------
    post_sample_data : pandas.DataFrame
        A DataFrame containing posterior samples for model parameters. Each column should correspond 
        to a parameter (e.g., "A", "tau", "mu_c", etc.).

    num_obs : int
        The number of observations (e.g., trials) to simulate for each resimulation. Typically matches 
        the size of the empirical dataset.

    simulator : object
        A simulator object with an `experiment(...)` method that accepts the relevant parameters 
        and returns simulated data in a tabular format (e.g., list of dicts or DataFrame-compatible structure). 
        The object may also have an attribute `sdr_fixed` which controls whether `sd_r` is passed explicitly.

    part : str or int
        A label identifying the participant for whom the resimulations are being generated.

    num_resims : int, optional
        The number of independent resimulation runs to perform. Default is 50.

    param_names : list of str, optional
        The list of parameter names to consider when filtering and passing values to the simulator. 
        These should match the columns in `post_sample_data`. Default is ["A", "tau", "mu_c", "mu_r", "b"].

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing all simulated trials across resimulations. Includes:
        - Simulated trial data from the `simulator`
        - A "num_resim" column indicating the resimulation index
        - A "participant" column identifying the source participant

    Notes:
    ------
    - Posterior samples with negative values are excluded before resimulation. The number of 
      excluded samples is tracked but not returned; consider logging or returning `excluded_samples` if needed.
    - The function assumes that enough valid (non-negative) samples are available to perform `num_resims`.
    - If `simulator.sdr_fixed` is not `None`, `sd_r` will not be passed as a parameter.

    """

    # convert to dict (allow differing number of samples per parameter)
    resim_samples = dict(post_sample_data)

    # count excluded samples
    excluded_samples = dict()

    excluded_samples['num_samples'] = post_sample_data.shape[0]
    excluded_samples["participant"] = part

    # exclude negative samples
    for k, dat in resim_samples.items():
        if k in param_names:
            samples = dat.values[dat.values >= 0]
            np.random.shuffle(samples)
            resim_samples[k] = samples

            excluded_samples[k] = dat.values[dat.values < 0].shape[0]

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
    """
    Computes delta plots from response time (RT) data across quantiles.

    This function calculates RT quantiles separately for different experimental conditions 
    (e.g., "congruent" vs "incongruent") within groups (e.g., participants), and derives 
    delta functions by computing the difference between conditions across quantiles. This 
    is commonly used in cognitive modeling and conflict processing research.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing trial-level data, including RTs, condition labels, 
        and grouping variables (e.g., participants).

    quantiles : array-like, optional
        A sequence of quantile values (between 0 and 1) at which to compute RTs. 
        Defaults to `np.arange(0, 1, 0.1)` (i.e., deciles excluding the 1.0 quantile).

    grouping_labels : list of str, optional
        Columns used to group the data before computing quantiles. These typically include 
        participant ID and condition label. Default is ["participant", "condition_label"].

    rt_var : str, optional
        Name of the column in `data` that contains response times. Default is "rt".

    congruency_name : str, optional
        Name of the column that contains condition labels such as "congruent" 
        and "incongruent". Default is "condition_label".

    Returns:
    --------
    pandas.DataFrame
        A DataFrame indexed by quantile, with columns:
        - 'congruent': RT quantile values for the congruent condition
        - 'incongruent': RT quantile values for the incongruent condition
        - 'delta': difference between incongruent and congruent RTs at each quantile
        - 'mean_qu': mean of incongruent and congruent RTs at each quantile

    Notes:
    ------
    - This function assumes that `congruency_name` contains exactly two levels: 
      "congruent" and "incongruent".
    - It reshapes the data to wide format to facilitate delta function calculation.
    - Output can be used to plot delta plots for visualization or modeling of conflict effects.

    """
    
    # compute quantiles 
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
    """
    Formats a list of parameter names for LaTeX-style labeling (e.g., for plotting).

    This function wraps each parameter name in LaTeX math mode formatting, optionally adding 
    a backslash prefix (`\\`) for specific Greek-like symbols (e.g., "tau", "mu_c", "mu_r"), 
    which are typically rendered as LaTeX commands (e.g., "\\tau").

    Parameters:
    -----------
    param_names : list of str
        A list of parameter names (e.g., ["A", "tau", "mu_c"]) to be formatted.

    Returns:
    --------
    list of str or str
        A list of LaTeX-formatted strings if the input contains multiple parameters,
        or a single formatted string if only one parameter is provided.

    Examples:
    ---------
    >>> param_labels(["A", "tau", "mu_c"])
    ['$A$', '$\\tau$', '$\\mu_c$']

    >>> param_labels(["tau"])
    '$\\tau$'

    Notes:
    ------
    - The function assumes that any parameter in ["tau", "mu_c", "mu_r"] should be interpreted 
      as a LaTeX symbol and prefixed with a backslash.
    - The returned strings can be used directly as axis labels in Matplotlib or other plotting libraries
      that support LaTeX-style rendering.
    """

    param_labels = []

    for p in param_names:

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        param_labels.append(suff + p + "$")

    if len(param_labels) <= 1:
        param_labels = param_labels[0]
        
    return param_labels


def cohens_d_samples(samples1, samples2, param_names, num_samples=1000, sharex=True, subj_id='Subject', hdi_color='black', hdi_alpha=1, x_prop=0.05, y_prop=0.85, text_rotation=0, zero_line=True, x_lower=-1.2, x_upper=1.2):
    """
    Computes and visualizes Cohen's d for paired posterior parameter samples across multiple participants.

    This function calculates standardized mean differences (Cohen's d) between two posterior sample sets 
    (e.g., from different experimental conditions) for each parameter of interest. The differences are 
    computed across participants for each Monte Carlo sample and summarized via KDE plots, including 
    posterior means and 95% highest density intervals (HDIs).

    Parameters:
    -----------
    samples1 : pandas.DataFrame
        Posterior samples from condition 1 (e.g., control), with one column per parameter and one row 
        per sample per participant. Must include a `subj_id` column.

    samples2 : pandas.DataFrame
        Posterior samples from condition 2 (e.g., experimental), formatted identically to `samples1`.

    param_names : list of str
        Names of the parameters for which Cohen's d should be computed.

    num_samples : int, optional
        Number of Monte Carlo samples to use for computing Cohen's d. Default is 1000.

    sharex : bool, optional
        Whether the x-axis should be shared across subplots. Default is True.

    subj_id : str, optional
        Column name identifying the subject or participant in both sample sets. Default is 'Subject'.

    hdi_color : str, optional
        Color used for the KDE line. Default is 'black'.

    hdi_alpha : float, optional
        Alpha transparency level for the filled KDE. Default is 1 (opaque).

    x_prop : float, optional
        Proportional x-position (in axis coordinates) for placing the mean d text label. Default is 0.05.

    y_prop : float, optional
        Proportional y-position (in axis coordinates) for placing the mean d text label. Default is 0.85.

    text_rotation : int or float, optional
        Rotation angle (in degrees) for the mean d text. Default is 0.

    zero_line : bool, optional
        Whether to draw a vertical line at d = 0 for visual reference. Default is True.

    x_lower : float, optional
        Lower bound of the x-axis for all subplots. Default is -1.2.

    x_upper : float, optional
        Upper bound of the x-axis for all subplots. Default is 1.2.

    Returns:
    --------
    data_d : pandas.DataFrame
        DataFrame containing Cohen's d values across all Monte Carlo samples for each parameter.

    fig : matplotlib.figure.Figure
        Figure containing the KDE plots for each parameter's standardized mean difference distribution.

    Notes:
    ------
    - Assumes the same number of participants and sample structure in both `samples1` and `samples2`.
    - Issues warnings if participant IDs do not match between samples.
    - Uses standard deviation of paired differences as the denominator for computing Cohen's d.
    - Uses seaborn for density visualization and matplotlib for figure layout.
    - This function is intended for paired comparison designs where within-subject parameter estimates are compared.

    Example:
    --------
    >>> cohens_d_samples(samples_control, samples_treatment, ["A", "tau", "mu_c"])
    """

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
        hdi_bounds = hdi(data_d[p].values, hdi_prob=0.95)

        # HDI as shaded region with a different, subtle color
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=True, alpha=0.3,linewidth=0)
        ax.axvspan(ax.get_xlim()[0], hdi_bounds[0], color='white', alpha=1)  # Left of HDI
        ax.axvspan(hdi_bounds[1], ax.get_xlim()[1], color='white', alpha=1)  # Right of HDI
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=False, alpha=1,linewidth=1)

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