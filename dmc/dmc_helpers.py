import pandas as pd
import numpy as np
import time
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Mapping, Sequence, Union, Dict, List, Any, Iterable, Hashable, Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy.typing as npt
from tqdm import tqdm


def hdi(
    samples: Sequence[float] | npt.NDArray[np.floating],
    hdi_prob: float = 0.95
) -> Tuple[float, float]:
    """
    Compute the Highest Density Interval (HDI) of a sample distribution.

    Parameters
    ----------
    samples : Sequence[float] or numpy.ndarray
        1D array-like object of posterior samples.
    hdi_prob : float
        The desired probability for the HDI (e.g., 0.95 for 95% HDI).

    Returns
    -------
    Tuple[float, float]
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

    hdi_min = float(sorted_samples[min_idx])
    hdi_max = float(sorted_samples[min_idx + interval_idx_inc])

    return hdi_min, hdi_max


def check_vars(data: pd.DataFrame,
               rt: str = None,
               accuracy: str = None,
               id_name: str = None,
               congruency: str = None):

    var_names = ['Reaction Times (rt=...)', 'Accuracy (accuracy=...)', 'the Identifier (id_name=...)', 'Congruency (congruency=...)']

    for i, var in enumerate([rt, accuracy, id_name, congruency]):
        if var is not None:
            if var not in set(data.columns):
                raise ValueError(f"Variable '{var}' does not exist in data. Please specify a valid name for {var_names[i]}.")

def check_congruency(
    data: pd.DataFrame,
    rt: str = None,
    congruency: str = None,
    output_coding_con="congruent",
    output_coding_inc="incongruent",
):
    check_vars(data=data, rt=rt, congruency=congruency, accuracy=None, id_name=None)

    if congruency is None:
        return data

    congruency_labels = set(data[congruency].dropna().unique())

    allowed = [
        {"congruent", "incongruent"},
        {0, 1},
        {"con", "inc"},
        {output_coding_con, output_coding_inc},
    ]
    if not any(congruency_labels == s for s in allowed):
        raise ValueError(
            f"Congruency variable is coded as {congruency_labels}. Please recode "
            f"'{congruency}' to 'congruent' / 'incongruent' before submitting data to this function."
        )

    # recode using map (no FutureWarning)
    if congruency_labels == {"con", "inc"}:
        mapping = {"con": output_coding_con, "inc": output_coding_inc}
        data[congruency] = data[congruency].map(mapping).astype("object")

        mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
        mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
        diff = mean_inc - mean_con

        warnings.warn(
            f"'{congruency}' has been recoded to con -> {output_coding_con} / inc -> {output_coding_inc}. "
            f"RT Difference between incongruent - congruent conditions: {diff}."
        )

    elif congruency_labels == {0, 1} and congruency_labels != {output_coding_con, output_coding_inc}:
        mapping = {0: output_coding_con, 1: output_coding_inc}
        data[congruency] = data[congruency].map(mapping).astype("object")

        mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
        mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
        diff = mean_inc - mean_con

        warnings.warn(
            f"'{congruency}' has been recoded to 0 -> {output_coding_con} / 1 -> {output_coding_inc}. "
            f"RT Difference between incongruent - congruent conditions: {diff}."
        )

    elif congruency_labels == {"congruent", "incongruent"}:
        mapping = {"congruent": output_coding_con, "incongruent": output_coding_inc}
        data[congruency] = data[congruency].map(mapping).astype("object")

        mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
        mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
        diff = mean_inc - mean_con

        if {output_coding_con, output_coding_inc} != {"congruent", "incongruent"}:
            warnings.warn(
                f"'{congruency}' has been recoded to congruent -> {output_coding_con} / incongruent -> {output_coding_inc}. "
                f"RT Difference between incongruent - congruent conditions: {diff}."
            )

    # final sanity check (runs for all cases)
    mean_con = data.loc[data[congruency] == output_coding_con, rt].mean()
    mean_inc = data.loc[data[congruency] == output_coding_inc, rt].mean()
    diff = mean_inc - mean_con
    if diff < 0:
        warnings.warn(
            f"RT Difference between incongruent - congruent conditions is negative: {diff}. "
            f"Please check the coding of congruency conditions."
        )

    return data


def format_empirical_data(
    data: pd.DataFrame,
    rt: str = None,
    accuracy: str = None,
    congruency: str = None,
) -> Dict[str, np.ndarray]:
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

    rt : str
        Column name for reaction time in empirical data set.
    accuracy : str
        Column name for accuracy in empirical data set.
    congruency : str
        Column name for congruency (coded as or 0 (congruent) /1 (incongruent)).

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

    data = check_congruency(data=data, rt=rt, congruency=congruency, output_coding_con=0, output_coding_inc=1)

    var_names = [rt, accuracy, congruency]
    
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


def fit_empirical_data(
    data: pd.DataFrame,
    approximator: Any,
    num_samples: int = 1000,
    id_name: str = "id",
    rt: str = None,
    accuracy: str = None,
    congruency: str = None,
) -> pd.DataFrame:
    """
    Samples posteriors for empirical data for each unique subject or group.

    This function iterates over unique identifiers in the input DataFrame (e.g., participants),
    formats their data appropriately, performs posterior sampling using the specified 
    approximator, and aggregates the results into a combined DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing empirical observations. Must include a column corresponding
        to `id_name` to distinguish between different units (e.g., participants).
    
    approximator : bayesflow.approximators.ContinuousApproximator
        A trained BayesFlow `ContinuousApproximator` object used to perform amortized 
        posterior inference. It must implement a `.sample(conditions, num_samples)` method,
        where `conditions` is a dictionary of formatted input data and `num_samples` 
        is the number of posterior samples to draw.

    id_name : str, optional
        The column name used to identify unique units in the data (e.g., "participant").
        Defaults to "id".

    rt : str
            Column name for reaction time in empirical data set.
    accuracy : str
        Column name for accuracy in empirical data set.
    congruency : str
        Column name for congruency (coded as or 0 (congruent) /1 (incongruent)).

    Returns:
    --------
    pandas.DataFrame
        A concatenated DataFrame containing posterior samples for all individuals.
        Includes:
        - Flattened posterior samples (one column per variable)
        - The participant/group identifier (`id_name`)
        - The sampling time for each individual (`sampling_time`)

    Notes:
    ------
    - This function assumes that the `format_empirical_data` function is available
      and correctly formats individual data into a dictionary suitable for the 
      approximator.
    - The `approximator` must support a `sample` method with arguments:
      `conditions` (dict) and `num_samples` (int).
    """
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        tqdm.write(f"{category.__name__}: {message}")

    warnings.showwarning = custom_warning_handler

    check_vars(data=data, rt=rt, accuracy=accuracy, congruency=congruency, id_name=id_name)

    # extract unique id labels
    ids=data[id_name].unique()

    list_data_samples=[]

    # iterate over participants
    for i in tqdm(range(0, len(ids)), desc="Sampling posteriors"):

        id = ids[i]
        
        # select participant data
        part_data = data[data[id_name]==id]
        
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            # format empirical data for approximator
            part_data = format_empirical_data(
                part_data,
                rt=rt,
                accuracy=accuracy,
                congruency=congruency
            )

            # If a warning occurred 
            for w in caught:
                tqdm.write(
                    f"[ID {id}] {w.category.__name__}: {w.message}"
                )

        # draw posterior samples with the given approximator
        start_time=time.time()
        samples = approximator.sample(conditions=part_data, num_samples=num_samples)
        end_time=time.time()
        
        # computing total sampling time
        sampling_time=end_time-start_time

        # reformat it back into a numpy array -> DataFrame
        samples_2d={k: v.flatten() for k, v in samples.items()}
        
        data_samples=pd.DataFrame(samples_2d)
        
        data_samples[id_name]=id
        data_samples["sampling_time"]=sampling_time
        
        list_data_samples.append(data_samples)

    # combine data frames from all participants
    data_samples_complete=pd.concat(list_data_samples)

    return data_samples_complete


def weighted_metric_sum(
    metrics_table: pd.DataFrame,
    weight_recovery: float = 1.0,
    weight_pc: float = 1.0,
    weight_sbc: float = 1.0,
) -> float:
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
    mt = metrics_table.copy()
    mt.iloc[1, :] = 1 - mt.iloc[1, :]

    # compute means across parameters
    metrics_means=mt.mean(axis=1)

    # decide on weights for each metric (Recovery, Posterior Contraction, SBC)
    metrics_weights=np.array([weight_recovery, weight_pc, weight_sbc])

    # compute weighted sum
    weighted_sum=np.dot(metrics_means, metrics_weights)
    
    return weighted_sum


def post_samples_to_df(post_samples):
    """
    Convert batched posterior samples into a long-format pandas DataFrame.

    This function takes posterior samples stored as a dictionary of NumPy arrays
    (e.g., as returned by a BayesFlow approximator) and converts them into a single
    concatenated pandas DataFrame. Each batch element (e.g., participant or dataset)
    is assigned a unique integer identifier via an `id` column.

    Parameters
    ----------
    post_samples : dict
        Dictionary of posterior samples. Each key corresponds to a model parameter
        name, and each value must be a NumPy array with shape
        `(n_ids, n_samples, n_dims)`, where:
        - `n_ids` is the number of independent units (e.g., participants),
        - `n_samples` is the number of posterior samples per unit,
        - `n_dims` is the parameter dimensionality (typically 1).

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame containing posterior samples with one row per sample.
        The DataFrame includes:
        - One column per parameter in `post_samples`
        - An integer column `id` identifying the originating unit

        The total number of rows is `n_ids × n_samples`.

    Notes
    -----
    - All parameter arrays in `post_samples` are assumed to have identical shapes
      in their first two dimensions (`n_ids`, `n_samples`).
    - Parameter arrays are flattened along the last dimension before insertion
      into the DataFrame.
    - The `id` column is zero-indexed and assigned in the order of the first
      dimension of the arrays.

    Examples
    --------
    >>> df = post_samples_to_df(post_samples)
    >>> df.head()
           A   tau   mu_c   mu_r     b  id
    0  0.45  0.32   1.12   0.98  0.21   0
    """

    ids = post_samples['A'].shape[0]

    lst_samples = []

    for id in range(0, ids):

        samples_2d = {k: v[id, :, :].flatten() for k, v in post_samples.items()}
        
        df_single = pd.DataFrame(samples_2d)

        df_single['id'] = id

        lst_samples.append(df_single)

    return pd.concat(lst_samples)

def resim_data_id(
    post_sample_data: Union[pd.DataFrame, Mapping[str, np.ndarray]],
    num_obs: int,
    simulator: Any,
    id: Union[str, int],
    id_name: Union[str, int] = 'id',
    num_resims: int = 50,
    param_names: Sequence[str] = ("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
    lower_bound: float = 0
) -> pd.DataFrame:
    """
    Resimulate trial-level data for one participant or observational unit from
    posterior parameter samples.

    This function takes posterior samples for a single unit, filters parameter
    draws below a lower bound, randomly shuffles the remaining valid draws within
    each parameter, and repeatedly calls ``simulator.experiment(...)`` to produce
    posterior predictive datasets.

    For each resimulation, one value per parameter is taken from the filtered
    sample arrays and passed to the simulator together with ``num_obs``. The
    resulting trial-level simulated data are concatenated across resimulations and
    annotated with a resimulation index (``num_resim``) and the supplied unit
    identifier (``id_name``).

    Parameters
    ----------
    post_sample_data : pandas.DataFrame or mapping of str to array-like
        Posterior samples for a single participant or unit. Each parameter should
        be stored under its own column/key. Typical parameter names are
        ``"A"``, ``"tau"``, ``"mu_c"``, ``"mu_r"``, ``"b"``, and ``"sd_r"``.

        If a pandas DataFrame is provided, columns are converted to a dictionary
        internally. Values are expected to support ``.values`` and ``.shape``.

    num_obs : int
        Number of observations (for example, trials) to simulate for each
        posterior predictive resimulation.

    simulator : object
        Simulator object providing an ``experiment(...)`` method. This method must
        accept the parameters listed in ``param_names`` together with
        ``num_obs=num_obs`` and return trial-level simulated data in a format that
        can be converted to a pandas DataFrame.

    id : str or int
        Identifier for the participant or observational unit for whom posterior
        predictive data are generated.

    id_name : str, default='id'
        Name of the identifier column to be added to the returned resimulated
        DataFrame.

    num_resims : int, default=50
        Number of posterior predictive datasets to generate.

    param_names : sequence of str, default=("A", "tau", "mu_c", "mu_r", "b", "sd_r")
        Names of the parameters to extract from ``post_sample_data`` and pass to
        the simulator.

    lower_bound : float, default=0
        Lower bound for valid posterior samples. Values strictly below this bound
        are excluded before resimulation.

    Returns
    -------
    resim_complete : pandas.DataFrame
        Trial-level simulated data concatenated across all resimulations. Includes
        the simulator output columns plus:

        - ``"num_resim"``: resimulation index from ``0`` to ``num_resims - 1``
        - ``id_name``: the supplied unit identifier

    n_excluded_samples : int
        Total number of posterior samples excluded across all parameters listed in
        ``param_names`` because they were below ``lower_bound``.

    n_all_samples : int
        Total number of posterior samples inspected across all parameters listed
        in ``param_names`` before exclusion.

    Notes
    -----
    - The function currently returns a 3-tuple:
      ``(resim_complete, n_excluded_samples, n_all_samples)``.
    """

    # convert to dict (allow differing number of samples per parameter)
    resim_samples = dict(post_sample_data)

    n_excluded_samples = 0
    n_all_samples = 0

    # exclude negative samples
    for k, dat in resim_samples.items():
        if k in param_names:
            samples = dat.values[dat.values >= lower_bound]
            np.random.shuffle(samples)
            resim_samples[k] = samples

            n_all_samples += dat.shape[0]
            n_excluded_samples += dat.shape[0] - samples.shape[0]

    list_resim_dfs = []

    # resimulate
    for i in range(num_resims):

        iteration_dict = {key: values[i] for key, values in resim_samples.items() if key in param_names}

        resim =  simulator.experiment(**iteration_dict | {'num_obs': num_obs})

        resim_df = pd.DataFrame(resim)
        
        resim_df["num_resim"] = i
        resim_df[id_name] = id
        
        list_resim_dfs.append(pd.DataFrame(resim_df))

    resim_complete = pd.concat(list_resim_dfs)

    return resim_complete, n_excluded_samples, n_all_samples

def resim_data(empirical_data: pd.DataFrame, 
               post_samples: pd.DataFrame,
               simulator,
               num_resims: int = 50,
               param_names: Sequence[str] = ("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
               rt: str = 'rt',
               id_name: str = 'id',
               congruency: str = 'congruency',
               simulator_congruency: str = 'conditions',
               simulator_congruency_coding: float = 0.0,
               simulator_incongruency_coding: float = 1.0,
               exclude_nonconvergents: bool = True,
               lower_bound: float = 0):
    
    """
    Perform posterior-predictive resimulations for each unit in an empirical dataset.

    This function loops over all unique identifiers in `empirical_data[id_name]`,
    determines the number of empirical observations per identifier, subsets the
    corresponding posterior parameter samples from `post_samples`, and calls
    `resim_data_id(...)` to generate resimulated trial-level data via `simulator`.

    After simulation, the function:
    1) removes non-convergent trials (defined as `rt == -1`),
    2) recodes the numeric condition codes in the `conditions` column into a
       human-readable congruency label column (`congruency`) using the mapping
       `{0.0: "congruent", 1.0: "incongruent"}`.

    Parameters
    ----------
    empirical_data : pandas.DataFrame
        Empirical trial-level dataset containing at least the identifier column
        `id_name`. The number of rows per identifier determines `num_obs` passed
        to the simulator.

    post_samples : pandas.DataFrame
        Long-format posterior samples containing at least the identifier column
        `id_name`. For each identifier, this function selects the subset
        `post_samples[post_samples[id_name] == part]` and passes it to
        `resim_data_id(...)`.

    rt: str,
        Name of the reaction time variable in `data.

    simulator : object
        A simulator instance compatible with `resim_data_id(...)` (typically
        exposing an `experiment(...)` method).

    id_name : str, optional
        Name of the identifier column used to match empirical units to posterior
        samples. Default is `'id'`.

    congruency : str, optional
        Name of the output column storing congruency labels derived from the
        numeric simulator_congruency column. Default is `'congruency'`.

    simulator_congruency : str
        Name of the congruency conditions variable as simulated by the simulator.

    simulator_congruency_coding : float
        values/ label of the congruent condition in the simulator_congruency variable. Default is `0.0`.

    simulator_incongruency_coding: float
        values/ label of the incongruent condition in the simulator_congruency variable. Default is `1.0`.

    exclude_nonconvergents: bool
        Indicates if nonconvergent trials (rt = -1) should be excluded. Default is `True`.

    Returns
    -------
    list[pandas.DataFrame]
        A list of per-identifier resimulated datasets. Each element is a
        trial-level DataFrame produced by `resim_data_id(...)`, filtered to remove
        `rt == -1` rows and augmented with a congruency label column
        (`congruency`).

    External Dependencies / Assumptions
    -----------------------------------
    - `resim_data_id(...)` must be defined in the surrounding scope and accept
      arguments compatible with:
        `resim_data_id(part_data_samples, num_obs, simulator, id, param_names=param_names)`
    - `param_names` must exist in the surrounding scope (global or closure).
    - The resimulated output is expected to contain columns:
        - `'rt'` (reaction time; used to filter non-convergents)
        - `'conditions'` (numeric condition codes; used for congruency mapping)

    Notes
    -----
    - If `post_samples` is missing entries for an identifier in `empirical_data`,
      the corresponding resimulation may be empty or raise an error inside
      `resim_data_id(...)` depending on its implementation.
    - The congruency mapping assumes exactly two condition codes: 0.0 and 1.0.
      If your simulator uses different coding, adjust the mapping accordingly.
    """

    check_vars(data=empirical_data, id_name=id_name, rt=rt, congruency=congruency)

    ids = empirical_data[id_name].unique()

    excluded_samples = 0
    n_all_samples = 0

    lst_data = []
    pbar = tqdm(
        range(len(ids)),
        desc=f"Resimulate {num_resims} data sets per ID",delay=1.0)

    for i in pbar:
        id = ids[i]

        num_obs = empirical_data[empirical_data[id_name] == id].shape[0]
        part_data_samples = post_samples[post_samples[id_name] == id]

        data_resimulated, n_excluded_samples_id, n_all_samples_id = resim_data_id(part_data_samples,
                                                                                  num_obs=num_obs,
                                                                                  num_resims=num_resims,
                                                                                  simulator=simulator,
                                                                                  id=id,
                                                                                  param_names=param_names,
                                                                                  lower_bound=lower_bound)

        excluded_samples += n_excluded_samples_id 
        n_all_samples += n_all_samples_id

        if exclude_nonconvergents:
            data_resimulated = data_resimulated[data_resimulated[rt] != -1]

        data_resimulated[congruency] = data_resimulated[simulator_congruency].map(
            {
                simulator_congruency_coding: "congruent",
                simulator_incongruency_coding: "incongruent",
            }
        )

        lst_data.append(data_resimulated)

        percentage_excluded_samples = f'{np.round((excluded_samples/n_all_samples)*100, 3)}%'

        pbar.set_description(f"Resimulate {num_resims} data sets per ID | Excluded Samples={percentage_excluded_samples}")

    return pd.concat(lst_data)

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

def smd_samples(
    samples1: pd.DataFrame,
    samples2: pd.DataFrame,
    param_names: List[str],
    num_samples: int = 1000,
    sharex: bool = True,
    id_name: str = 'id',
    hdi_color: str = 'white',
    hdi_alpha: float = 1.0,
    x_prop: float = 0.05,
    y_prop: float = 0.85,
    zero_line: bool = True,
    x_lower: float = -1.2,
    x_upper: float = 1.2,
    fontsize: int = 15,
    fontsize_ticklabels: int = 12,
    fontsize_label: int = 15,
    fontsize_axis_labels: int = 15,
    figsize: Tuple[float, float] = (15.0, 3.0),
    supxlabel: str = 'Standardized Mean Difference $d_i$'
) -> Tuple[pd.DataFrame, Figure]:
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
        per sample per participant. Must include a `id_name` column.

    samples2 : pandas.DataFrame
        Posterior samples from condition 2 (e.g., experimental), formatted identically to `samples1`.

    param_names : list of str
        Names of the parameters for which Cohen's d should be computed.

    num_samples : int, optional
        Number of Monte Carlo samples to use for computing Cohen's d. Default is 1000.

    sharex : bool, optional
        Whether the x-axis should be shared across subplots. Default is True.

    id_name : str, optional
        Column name identifying the subject or participant in both sample sets. Default is 'id'.

    hdi_color : str, optional
        Color used for the KDE line. Default is 'white'.

    hdi_alpha : float, optional
        Alpha transparency level for the filled KDE. Default is 1 (opaque).

    x_prop : float, optional
        Proportional x-position (in axis coordinates) for placing the mean d text label. Default is 0.05.

    y_prop : float, optional
        Proportional y-position (in axis coordinates) for placing the mean d text label. Default is 0.85.

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
    >>> smd_samples(samples_control, samples_treatment, ["A", "tau", "mu_c"])
    """

    num_params = len(param_names)
    cohens_ds = np.ones((num_samples,num_params))

    parts = samples1[id_name].unique()

    # deterministic draw index within each participant
    samples1 = samples1.copy()
    samples2 = samples2.copy()

    samples1["sample_id"] = samples1.groupby(id_name).cumcount()
    samples2["sample_id"] = samples2.groupby(id_name).cumcount()

    # choose the same draw indices for both conditions
    draws = np.random.choice(num_samples, size=num_samples, replace=False)

    samples1 = samples1[samples1["sample_id"].isin(draws)]
    samples2 = samples2[samples2["sample_id"].isin(draws)]


    for j,p in enumerate(param_names):
        for i in range(0, num_samples):
            # control condition
            m1 = samples1[samples1['sample_id'] == i][p]
            #m1 = m1[~np.isnan(m1)]

            # experimental manipulation
            m2 = samples2[samples2['sample_id'] == i][p]
            #m2 = m2[~np.isnan(m2)]

            if set(samples1[samples1['sample_id'] == i][id_name].unique()) != set(parts):
                warnings.warn(f'Participants in sub sample 1 and sample id {i} are not identical to all participants!')
            
            if set(samples2[samples2['sample_id'] == i][id_name].unique()) != set(parts):
                warnings.warn(f'Participants in sub sample 2 and sample id {i} are not identical to all participants!')

            if m1.shape[0] != parts.shape[0] or m2.shape[0] != parts.shape[0]:
                warnings.warn(f'Mismatch in number of entries in sample id {i}')

            m1 = m1.values
            m2 = m2.values

            d = np.mean(m1) - np.mean(m2)

            diff = m1 - m2
            sd = np.std(diff, ddof=1)
            mean_d = np.nan if sd == 0 else d / sd

            cohens_ds[i,j] = mean_d

    data_d = pd.DataFrame(cohens_ds, columns = param_names)

    
    fig, axes = plt.subplots(1, len(param_names), figsize=figsize, sharex=sharex)

    if len(param_names) == 1:
        axes = [axes]

    for p, ax in zip(param_names, axes):

        ax.set_xlim(x_lower, x_upper)

        post_mean = np.mean(data_d[p])
        ax.axvline(x=post_mean, color='black', linestyle='--', linewidth=1)

        if zero_line:
            ax.axvline(x=0, color='red', linestyle='-', linewidth=1)

        #ax.set_xlim(x_lower, x_upper)
        hdi_bounds = hdi(data_d[p].values, hdi_prob=0.95)

        # HDI as shaded region with a different, subtle color
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=True, alpha=0.3,linewidth=0)
        ax.axvspan(ax.get_xlim()[0], hdi_bounds[0], color=hdi_color, alpha= hdi_alpha)  # Left of HDI
        ax.axvspan(hdi_bounds[1], ax.get_xlim()[1], color=hdi_color, alpha= hdi_alpha)  # Right of HDI
        sns.kdeplot(data=data_d, x=p, ax=ax, color='#132a70', fill=False, alpha=1,linewidth=1)

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        label = suff + p + "$"

        ax.set_title(label, fontsize=fontsize)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=fontsize_ticklabels)  


        if p == 'A':
            ax.set_ylabel('Density', fontsize=fontsize_axis_labels)
        else:
            ax.set_ylabel('')

        ymax = ax.get_ylim()[1]
        xmin = ax.get_xlim()[0]
        xmax = ax.get_xlim()[1]

        x_range = xmax-xmin

        ax.text(xmin + x_range*x_prop, ymax*y_prop, '$d = $' + str(round(post_mean, 2)), fontsize=fontsize_label, color='black', rotation=0)
    
    fig.supxlabel(supxlabel, fontsize=fontsize)
    fig.tight_layout()

    return data_d, fig



def format_sim_data(
    sim_data: Dict[str, np.ndarray],
    congruency_coding: int = 0,
    only_convergents: bool = True,
    id_name: str = 'id'
) -> pd.DataFrame:
    """
    Format simulated behavioral data into a long-format pandas DataFrame.

    This function takes batched simulation output (reaction times, accuracy,
    and condition codes) and converts it into a single concatenated DataFrame
    suitable for downstream statistical analysis or visualization (e.g. compute_stats, plot_stats).

    Parameters
    ----------
    sim_data : Dict[str, np.ndarray]
        Dictionary containing simulation outputs with the following keys:
        - 'rt': Reaction times, shape (batch_size, n_trials, 1)
        - 'accuracy': Accuracy values, shape (batch_size, n_trials, 1)
        - 'conditions': Condition codes, shape (batch_size, n_trials, 1)
    congruency_coding : int, optional
        Integer code indicating a congruent condition in `conditions`.
        All other values are treated as incongruent. Default is 0.
    only_convergents : bool, optional
        If True, remove trials with reaction time equal to -1,
        which are assumed to represent non-convergent simulations.
        Default is True.
    id_name : str, optional
        Variable name for data set identifier. Default to 'id',

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per trial and the following columns:
        - 'rt': Reaction time
        - 'accuracy': Accuracy value
        - 'conditions': Condition code
        - 'id': Batch index
        - 'congruency': 'congruent' or 'incongruent'
        - 'accuracy_name': 'correct' or 'incorrect'
    """
    batch_size: int = sim_data['rt'].shape[0]

    behav_keys = ['rt', 'accuracy', 'conditions']
    behav_data: Dict[str, np.ndarray] = {k: sim_data[k] for k in behav_keys}

    df_list = []
    rt_var = 'rt'

    for i in range(batch_size):
        stacked = np.stack(
            (
                behav_data['rt'][i, :, :],
                behav_data['accuracy'][i, :, :],
                behav_data['conditions'][i, :, :]
            ),
            axis=1
        )[:, :, 0]

        df_single = pd.DataFrame(stacked, columns=[rt_var, 'accuracy', 'conditions'])
        df_single[id_name] = i

        df_single['congruency'] = [
            'congruent' if x == congruency_coding else 'incongruent'
            for x in df_single['conditions']
        ]
        df_single['accuracy_name'] = [
            'correct' if x == 1.0 else 'incorrect'
            for x in df_single['accuracy']
        ]

        df_list.append(df_single)

    df_complete = pd.concat(df_list, ignore_index=True)

    if only_convergents:
        df_complete = df_complete[df_complete[rt_var] != -1]

    return df_complete


def compute_stats_ppc(
    data: pd.DataFrame,
    id_name: str = "id",
    draw_name: Optional[str] = 'num_resim',
    n_rt_bins: int = 5,
    rt: str = "rt",
    accuracy: str = "accuracy",
    congruency: str = "congruency",
    quantiles: Union[np.ndarray, Sequence[float]] = np.arange(0.1, 1.0, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    """
    Compute CAF, CDF-style quantile, and delta-function summaries for empirical
    or posterior predictive check (PPC) data.

    If ``draw_name`` is None, the function returns participant-level summaries.

    If ``draw_name`` is provided, the function first computes participant-level
    summaries within each draw and then averages these summaries across
    participants *within draw*, yielding one summary curve per draw. This is the
    appropriate structure for posterior predictive checks.

    Parameters
    ----------
    data : pandas.DataFrame
        Trial-level data. Required columns are:

        - ``id_name``: participant or unit identifier
        - ``rt``: reaction time
        - ``accuracy``: accuracy indicator (1 = correct, 0 = incorrect)
        - ``congruency``: condition label

        If ``draw_name`` is not None, the column named by ``draw_name`` must also
        be present and identify PPC draws / resimulations.

    id_name : str, default='id'
        Column name identifying participants or independent units.

    draw_name : str or None, default='num_resim'
        Column name identifying PPC draws / resimulations. If provided, summaries
        are computed within each draw and then aggregated across participants
        within draw. If None, participant-level summaries are returned directly.

    n_rt_bins : int, default=5
        Number of quantile-based RT bins used for CAF computation. RT bins are
        computed separately within each grouping cell:

        - ``id_name × congruency`` if ``draw_name`` is None
        - ``draw_name × id_name × congruency`` if ``draw_name`` is provided

    rt : str, default='rt'
        Column name containing reaction times.

    accuracy : str, default='accuracy'
        Column name containing accuracy values (1 = correct, 0 = incorrect).

    congruency : str, default='congruency'
        Column name indicating congruency condition.

    quantiles : array-like, default=np.arange(0.1, 1.0, 0.1)
        Quantile levels used to compute RT quantiles for CDF-style and delta
        summaries. Quantiles are computed using correct trials only
        (``accuracy == 1``).

    Returns
    -------
    caf_data : pandas.DataFrame
        CAF summaries.

        If ``draw_name`` is None:
            Participant-level CAF summaries with columns including
            ``id_name``, ``congruency``, ``rt_bin``, and ``accuracy``.

        If ``draw_name`` is provided:
            Draw-level CAF summaries averaged across participants within draw, with
            columns including ``draw_name``, ``congruency``, ``rt_bin``, and
            ``accuracy``.

    cdf_data : pandas.DataFrame
        Long-format CDF-style quantile summaries.

        If ``draw_name`` is None:
            Participant-level quantile summaries with columns including
            ``id_name``, ``quantile``, ``congruency``, and ``rt``.

        If ``draw_name`` is provided:
            Draw-level quantile summaries averaged across participants within draw,
            with columns including ``draw_name``, ``congruency``, ``quantile``,
            and ``rt``.

    delta_data : pandas.DataFrame
        Delta-function summaries derived from correct-trial quantiles.

        If ``draw_name`` is None:
            Participant-level delta summaries with columns including
            ``id_name``, ``quantile``, ``congruent``, ``incongruent``, ``delta``,
            and ``mean_qu``.

        If ``draw_name`` is provided:
            Draw-level delta summaries averaged across participants within draw,
            with columns including ``draw_name``, ``quantile``, ``congruent``,
            ``incongruent``, ``delta``, and ``mean_qu``.

    Raises
    ------
    KeyError
        If one or more required columns are missing from ``data``.

    ValueError
        If the congruency recoding does not produce the expected levels
        ``'congruent'`` and ``'incongruent'`` after pivoting, or if RT binning via
        ``pandas.qcut`` cannot be computed.

    Notes
    -----
    - CDF-style summaries and delta functions are computed from correct trials only.
    - CAF summaries are computed from all trials.
    - RT bins for CAFs are formed within grouping cells using
    ``pandas.qcut(..., duplicates="drop")``; therefore, some groups may yield
    fewer than ``n_rt_bins`` bins if too many duplicate RT values are present.
    """

    required = [id_name, rt, accuracy, congruency]
    if draw_name is not None:
        required.append(draw_name)

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = data.copy()

    if "check_congruency" in globals():
        df = check_congruency(
            data=df,
            rt=rt,
            congruency=congruency,
            output_coding_con="congruent",
            output_coding_inc="incongruent",
        )

    df[rt] = pd.to_numeric(df[rt], errors="coerce")
    df = df.dropna(subset=[rt, accuracy, congruency, id_name])

    group_base = [id_name, congruency]
    if draw_name is not None:
        group_base = [draw_name] + group_base

    # ------------------------------------------------------------
    # 1) DELTA / CDF: participant-level within draw
    # ------------------------------------------------------------
    correct_df = df[df[accuracy] == 1].copy()

    delta_subject = (
        correct_df.groupby(group_base, observed=False)[rt]
        .quantile(quantiles)
        .reset_index()
    )

    quantile_col = delta_subject.columns[-2]
    delta_subject = delta_subject.rename(columns={quantile_col: "quantile"})

    delta_subject = (
        delta_subject.pivot(
            index=([draw_name] if draw_name is not None else []) + [id_name, "quantile"],
            columns=congruency,
            values=rt,
        )
        .reset_index()
    )

    expected_cols = {"congruent", "incongruent"}
    if not expected_cols.issubset(delta_subject.columns):
        raise ValueError(
            f"Expected congruency levels {expected_cols}, but got "
            f"{set(delta_subject.columns)} after pivot."
        )

    delta_subject = delta_subject.assign(
        delta=lambda x: x["incongruent"] - x["congruent"],
        mean_qu=lambda x: (x["incongruent"] + x["congruent"]) / 2,
    )

    cdf_subject = pd.melt(
        delta_subject,
        id_vars=([draw_name] if draw_name is not None else []) + [id_name, "quantile"],
        value_vars=["congruent", "incongruent"],
        var_name=congruency,
        value_name=rt,
    )

    # ------------------------------------------------------------
    # 2) CAF: participant-level within draw
    # ------------------------------------------------------------

    try:
        df["rt_bin"] = (
            df.groupby(group_base, observed=False)[rt]
            .transform(lambda x: pd.qcut(x, q=n_rt_bins, labels=False, duplicates="drop"))
        )

        df = df.dropna(subset=["rt_bin"]).copy()
        df["rt_bin"] = df["rt_bin"].astype(int)
        
    except ValueError as e:
        raise ValueError(f"Could not compute RT bins with qcut: {e}") from e

    caf_subject = (
        df.groupby(group_base + ["rt_bin"], observed=False)[accuracy]
        .mean()
        .reset_index()
    )

    # ------------------------------------------------------------
    # 3) If no draw column: return participant-level summaries
    # ------------------------------------------------------------
    if draw_name is None:
        return caf_subject, cdf_subject, delta_subject

    # ------------------------------------------------------------
    # 4) PPC aggregation:
    #    average across participants within draw
    # ------------------------------------------------------------
    caf_data = (
        caf_subject.groupby([draw_name, congruency, "rt_bin"], observed=False)[accuracy]
        .mean()
        .reset_index()
    )

    cdf_data = (
        cdf_subject.groupby([draw_name, congruency, "quantile"], observed=False)[rt]
        .mean()
        .reset_index()
    )

    delta_data = (
        delta_subject.groupby([draw_name, "quantile"], observed=False)[
            ["congruent", "incongruent", "delta", "mean_qu"]
        ]
        .mean()
        .reset_index()
        .sort_values([draw_name, "mean_qu"])
    )

    return caf_data, cdf_data, delta_data


def compute_stats(
    data: pd.DataFrame,
    id_name: str = "id",
    n_rt_bins: int = 5,
    rt: str = 'rt',
    accuracy: str = 'accuracy',
    congruency: str = "congruency",
    quantiles: Union[np.ndarray, Sequence[float]] = np.arange(0.1, 1.0, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute distributional summary statistics for reaction-time (RT) data,
    producing inputs suitable for CAF, CDF-style quantile, and Δ-function plots.

    This function derives three DataFrames:

    1. **CAF data (`caf_data`)**:
    Mean accuracy per RT bin, stratified by ``id_name × congruency × rt_bin``.
    RT bins are computed separately within each ``id_name × congruency`` group
    using quantile-based binning.

    2. **CDF data (`cdf_data`)**:
    Long-format representation of correct-trial RT quantiles from `delta_data`,
    with columns ``[id_name, quantile, congruency, rt]`` suitable for plotting
    CDF-style quantile curves.

    3. **Δ-function data (`delta_data`)**:
    Quantiles of RT computed *only on correct trials* (``accuracy == 1``) for each
    ``id_name × congruency`` group, then pivoted to wide format with separate
    columns per congruency level (expected: ``'congruent'`` and ``'incongruent'``).
    It additionally computes:

    - ``delta = incongruent - congruent``
    - ``mean_qu = (incongruent + congruent) / 2``

    Parameters
    ----------
    data : pandas.DataFrame
        Trial-level (long-format) data containing RTs and accuracy. Required columns:

        - ``rt`` : float
            Reaction time (typically seconds).
        - ``accuracy`` : int | bool | float
            Trial accuracy indicator. Trials with ``accuracy == 1`` are treated as
            correct for Δ-function and CDF quantiles.
        - ``id_name`` : hashable
            Identifier for subject/session/batch.
        - ``congruency`` : str-like / categorical
            Congruency label. The Δ-function computation assumes that the pivot will
            yield columns named ``'congruent'`` and ``'incongruent'``.

    id_name : str, default='id'
        Column name identifying independent units (e.g., participant, session, batch).

    n_rt_bins : int, default=5
        Number of quantile bins used to discretize RTs for the CAF computation
        within each ``id_name × congruency`` group.

    rt : str, default='rt'
        Column name of the reaction time variable.

    accuracy : str, default='accuracy'
        Column name of the accuracy variable (1 = correct, 0 = incorrect).

    congruency : str, default='congruency'
        Column name indicating congruency condition. For downstream computations,
        the values are expected to include levels that pivot to columns named
        ``'congruent'`` and ``'incongruent'``.

    quantiles : numpy.ndarray or Sequence[float], default=np.arange(0.1, 1.0, 0.1)
        Quantile levels at which to compute RT quantiles for correct trials.

    Returns
    -------
    caf_data : pandas.DataFrame
        DataFrame containing conditional accuracy values per RT bin. Expected columns:

        - ``id_name``
        - ``congruency``
        - ``rt_bin`` : int
        - ``accuracy`` : float

    cdf_data : pandas.DataFrame
        Long-format DataFrame containing correct-trial RT quantiles. Expected columns:

        - ``id_name``
        - ``quantile`` : float
        - ``congruency`` : str
        - ``rt`` : float

    delta_data : pandas.DataFrame
        Wide-format DataFrame with per-``id_name`` quantiles for each congruency level,
        plus derived columns ``delta`` and ``mean_qu``. Expected columns include:

        - ``id_name``
        - ``quantile`` : float
        - ``congruent`` : float
        - ``incongruent`` : float
        - ``delta`` : float
        - ``mean_qu`` : float

    Raises
    ------
    KeyError
        If required columns are missing from ``data``.

    ValueError
        If the required congruency levels do not produce ``'congruent'`` and
        ``'incongruent'`` columns after pivoting.

    Notes
    -----
    The function operates on copies of the input data and does not modify the
    original DataFrame in-place.

    RT bins for CAF computation are formed separately within each
    ``id_name × congruency`` group using ``pandas.qcut(..., duplicates="drop")``.
    As a result, some groups may yield fewer than ``n_rt_bins`` bins when too many
    duplicate RT values are present.
    """

    check_vars(data=data, rt=rt, accuracy=accuracy, congruency=congruency, id_name=id_name)

    data = data.copy()
    data[rt] = pd.to_numeric(data[rt], errors="coerce")
    data = data.dropna(subset=[id_name, congruency, rt, accuracy]).copy()

    data = check_congruency(data=data, rt=rt, congruency=congruency, output_coding_con='congruent', output_coding_inc='incongruent')

    data[rt] = pd.to_numeric(data[rt], errors="coerce")

    delta_data = (
    data.loc[data[accuracy] == 1]
    .groupby([id_name, congruency])[rt]
    .quantile(quantiles)
    .reset_index()
    )

    quantile_col = [c for c in delta_data.columns if c not in [id_name, congruency, rt]][0]
    delta_data = delta_data.rename(columns={quantile_col: "quantile"})

    delta_data = (
        delta_data
        .pivot(index=[id_name, "quantile"], columns=congruency, values=rt)
        .reset_index()
    )

    expected_cols = {"congruent", "incongruent"}
    if not expected_cols.issubset(delta_data.columns):
        raise ValueError(
            f"Expected congruency levels {expected_cols}, got {set(delta_data.columns)}"
        )

    delta_data = delta_data.assign(
        delta=lambda df: df["incongruent"] - df["congruent"],
        mean_qu=lambda df: (df["incongruent"] + df["congruent"]) / 2,
    )

    df = data.copy()

    df["rt_bin"] = (
        df.groupby([id_name, congruency])[rt]
        .transform(lambda x: pd.qcut(x, q=n_rt_bins, labels=False, duplicates="drop"))
    )

    df = df.dropna(subset=["rt_bin"]).copy()
    df["rt_bin"] = df["rt_bin"].astype(int)

    caf_data = (
        df.groupby([id_name, congruency, "rt_bin"], observed=False)[accuracy]
        .mean()
        .reset_index()
    )

    cdf_data = pd.melt(
        delta_data,
        id_vars=[id_name, "quantile"],
        value_vars=["congruent", "incongruent"],
        var_name=congruency,
        value_name=rt,
    )

    return caf_data, cdf_data, delta_data


def plot_stats(
    caf_data: pd.DataFrame,
    cdf_data: pd.DataFrame,
    delta_data: pd.DataFrame,
    id_name: str = "id",
    rt: str = 'rt',
    congruency: str = "congruency",
    individual_deltas: bool = False,
    individual_cafs: bool = False,
    individual_cdfs: bool = False,
    delta_ylim: Optional[Tuple[float, float]] = None,
    delta_xlim: Optional[Tuple[float, float]] = None,
    cdf_xlim: Optional[Tuple[float, float]] = None,
    fontsize: int = 14,
    fontsize_axes: int = 14,
    fontsize_ticklabels: int = 10,
    fontsize_legend: int = 12,
    legend: bool = True,
    new_plot: bool = True,
    hue_order: Sequence[str] = ("congruent", "incongruent"),
    palette: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    linewidth: float = 0.75,
    fig: Optional[Figure] = None,
    axes: Optional[Sequence[Axes]] = None,
    individual_alpha: float = 0.1,
    individual_linewidth: float = 0.25,
    linestyle: str = "--",
    marker: str = "o",
    markersize: float = 5,
    markeredgecolor: str = "none"
    ):

    """
    Plot CAF, CDF-style quantile curves, and Δ-function in a 1×3 layout.

    This function visualizes three standard distributional summaries commonly used
    in conflict-task analyses:

    1. **CAF (Conditional Accuracy Function)**:
       Mean accuracy as a function of RT bin, separated by congruency condition.

    2. **CDF-style quantile plot**:
       Mean RT quantiles as a function of cumulative probability, separated by
       congruency condition. 

    3. **Δ-function**:
       Difference between incongruent and congruent RT quantiles as a function of
       the mean RT at each quantile.

    Optionally, participant-level ("individual") curves can be overlaid for each
    panel, allowing visualization of variability across units (e.g., subjects).

    Parameters
    ----------
    caf_data : pandas.DataFrame
        DataFrame containing CAF summaries. Expected columns include:

        - ``"rt_bin"`` : RT bin index
        - ``"accuracy"`` : mean accuracy per bin
        - ``congruency`` : condition label
        - ``id_name`` : identifier (if individual CAFs are plotted)

    cdf_data : pandas.DataFrame
        Long-format DataFrame containing RT quantile summaries. Expected columns:

        - ``"quantile"`` : cumulative probability
        - ``rt`` : RT values
        - ``congruency`` : condition label
        - ``id_name`` : identifier (if individual CDFs are plotted)

    delta_data : pandas.DataFrame
        DataFrame containing Δ-function summaries. Expected columns include:

        - ``"quantile"``
        - ``"mean_qu"`` : mean RT across conditions at each quantile
        - ``"delta"`` : incongruent minus congruent RT difference
        - ``id_name`` : identifier (if individual Δ-functions are plotted)

    id_name : str, optional
        Column name identifying participants or independent units
        (default: ``"id"``).

    rt : str, optional
        Column name for RT values in ``cdf_data`` (default: ``"rt"``).

    congruency : str, optional
        Column name indicating congruency condition (default: ``"congruency"``).

    individual_deltas : bool, optional
        If True, overlay participant-level Δ-functions (default: False).

    individual_cafs : bool, optional
        If True, overlay participant-level CAF curves (default: False).

    individual_cdfs : bool, optional
        If True, overlay participant-level CDF-style curves (default: False).

    delta_ylim : tuple of float, optional
        Y-axis limits for the Δ-function subplot.

    delta_xlim : tuple of float, optional
        X-axis limits for the Δ-function subplot.

    cdf_xlim : tuple of float, optional
        X-axis limits for the CDF subplot.

    fontsize : int, optional
        Font size for subplot titles (default: 14).

    fontsize_axes : int, optional
        Font size for axis labels (default: 14).

    fontsize_ticklabels : int, optional
        Font size for tick labels (default: 10).

    fontsize_legend : int, optional
        Font size for the legend (default: 12).

    legend : bool, optional
        If True, display a legend in the CAF panel (default: True).

    new_plot : bool, optional
        If True, create a new figure. If False, draw into provided ``fig`` and
        ``axes`` (default: True).

    hue_order : sequence of str, optional
        Order of condition levels for plotting
        (default: ``("congruent", "incongruent")``).

    palette : Mapping[str, str], optional
        Color mapping for conditions
        (default: ``{"congruent": "#132a70", "incongruent": "#FF6361"}``).

    linewidth : float, optional
        Line width for aggregated curves (default: 0.75).

    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into if ``new_plot`` is False.

    axes : sequence of matplotlib.axes.Axes, optional
        Existing axes (length 3) to draw into if ``new_plot`` is False.

    individual_alpha : float, optional
        Alpha value for participant-level curves (default: 0.1).

    individual_linewidth : float, optional
        Line width for participant-level curves (default: 0.25).

    linestyle : str, optional
        Line style for aggregated curves (default: ``"--"``).

    marker : str, optional
        Marker style for aggregated curves (default: ``"o"``).

    markersize : float, optional
        Marker size (default: 5).

    markeredgecolor : str, optional
        Marker edge color (default: ``"none"``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.

    axes : sequence of matplotlib.axes.Axes
        The axes for the CAF, CDF-style quantile plot, and Δ-function.

    Examples
    --------
    >>> fig, axes = plot_stats(caf_data, cdf_data, delta_data)

    >>> fig, axes = plot_stats(
    ...     caf_data,
    ...     cdf_data,
    ...     delta_data,
    ...     individual_cafs=True,
    ...     individual_cdfs=True,
    ...     individual_deltas=True,
    ...     cdf_xlim=(0.25, 1.0),
    ...     delta_xlim=(0.3, 1.0),
    ...     delta_ylim=(-0.1, 0.2),
    ... )
    """


    mean_data_emp = (
        cdf_data.groupby(["quantile", congruency], observed=False)["rt"]
        .mean()
        .reset_index()
    )

    if new_plot or fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # -------------------------
    # CAF
    # -------------------------

    if individual_cafs:
        sns.lineplot(
            data=caf_data,
            x="rt_bin",
            y="accuracy",
            hue=congruency,
            estimator=None,
            units=id_name,
            ax=axes[0],
            hue_order=hue_order,
            palette=palette,
            linestyle='--',
            alpha=individual_alpha,
            markeredgecolor=markeredgecolor,
            errorbar=None,
            linewidth=individual_linewidth,
            legend=None,
        )

    sns.lineplot(
        data=caf_data,
        x="rt_bin",
        y="accuracy",
        hue=congruency,
        ax=axes[0],
        hue_order=hue_order,
        palette=palette,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        errorbar=None,
        linewidth=linewidth,
        legend=legend,
    )

    axes[0].set_ylim(0, 1)
    axes[0].set_title("CAF", fontsize=fontsize)
    axes[0].set_ylabel("CAF", fontsize=fontsize_axes)
    axes[0].set_xlabel("Bins", fontsize=fontsize_axes)

    # -------------------------
    # CDF
    # -------------------------
    if individual_cdfs:

        sns.lineplot(
            data=cdf_data,
            x=rt,
            y="quantile",
            hue=congruency,
            ax=axes[1],
            hue_order=hue_order,
            estimator=None,
            units=id_name,
            palette=palette,
            linestyle='--',
            alpha=individual_alpha,
            markeredgecolor=markeredgecolor,
            linewidth=individual_linewidth,
            legend=False,
        )

    sns.lineplot(
        data=mean_data_emp,
        x=rt,
        y="quantile",
        hue=congruency,
        ax=axes[1],
        hue_order=hue_order,
        palette=palette,
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        linewidth=linewidth,
        legend=False,
    )

    axes[1].set_title("CDF", fontsize=fontsize)
    axes[1].set_ylabel("Cumulative Density", fontsize=fontsize_axes)
    axes[1].set_xlabel("RT[s]", fontsize=fontsize_axes)

    # -------------------------
    # Delta
    # -------------------------
    delta_bins_emp = (
        delta_data.groupby("quantile", observed=False)[["mean_qu", "delta"]]
        .agg(
            mean_qu=("mean_qu", "mean"),
            delta=("delta", "mean"),
            sd_delta=("delta", "std"),
        )
        .reset_index()
        .sort_values("mean_qu")
    )

    if individual_deltas:
        sns.lineplot(
            data=delta_data,
            x="mean_qu",
            y="delta",
            hue=id_name,
            ax=axes[2],
            linewidth=individual_linewidth,
            alpha=individual_alpha,
            legend=False,
        )

    sns.lineplot(
        data=delta_bins_emp,
        x="mean_qu",
        y="delta",
        ax=axes[2],
        color="black",
        linestyle=linestyle,
        marker=marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        linewidth=linewidth,
        legend=False,
    )

    axes[2].set_ylabel(r"$\Delta$", fontsize=fontsize_axes)
    axes[2].set_xlabel("RT[s]", fontsize=fontsize_axes)
    axes[2].set_title(r"$\Delta$-Function", fontsize=fontsize)

    # -------------------------
    # Limits
    # -------------------------
    if cdf_xlim is not None:
        axes[1].set_xlim(cdf_xlim)

    if delta_ylim is not None:
        axes[2].set_ylim(delta_ylim)

    if delta_xlim is not None:
        axes[2].set_xlim(delta_xlim)

    # -------------------------
    # Legend + ticks
    # -------------------------
    if legend:
        axes[0].legend(
            title="",
            loc="lower right",
            fontsize=fontsize_legend,
            frameon=False,
        )

    for ax in axes:
        ax.tick_params(axis="x", labelsize=fontsize_ticklabels)
        ax.tick_params(axis="y", labelsize=fontsize_ticklabels)

    fig.tight_layout()
    return fig, axes


def plot_fit(
    caf_data: pd.DataFrame,
    cdf_data: pd.DataFrame,
    delta_data: pd.DataFrame,
    caf_data_emp: pd.DataFrame,
    cdf_data_emp: pd.DataFrame,
    delta_data_emp: pd.DataFrame,
    congruency: str = "congruency",
    congruency_emp: str = "congruency",
    n_delta_bins: int = 10,
    delta_ylim: Optional[Tuple[float, float]] = None,
    delta_xlim: Optional[Tuple[float, float]] = None,
    cdf_xlim: Optional[Tuple[float, float]] = None,
    fontsize: int = 14,
    fontsize_axes: int = 14,
    fontsize_ticklabels: int = 10,
    fontsize_legend: int = 12,
    legend: bool = True,
    new_plot: bool = True,
    caf_errorbars: Optional[object] = None,
    hue_order: Sequence[str] = ("congruent", "incongruent"),
    palette_emp: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    palette_model: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    delta_linestyle_model: str = "-",
    caf_linestyle_model: str = "-",
    cdf_linestyle_model: str = "-",
    linewidth: float = 0.5,
    fig: Optional[Figure] = None,
    axes: Optional[Sequence[Axes]] = None,
    alpha: float = 0.05,
    plot_individual_deltas: bool = False,
    id_individual: str = 'id'):
    """
    Plot model and empirical CAFs, CDFs, and Δ-function in a 1×3 subplot layout.

    This function creates three panels:
    1. CAF (Conditional Accuracy Function)
    2. CDF (Cumulative Distribution Function of RTs)
    3. Δ-function (delta between conditions as a function of RT)

    Parameters
    ----------
    delta_data : pandas.DataFrame
        Model delta data with at least the columns:
        ['quantile', 'mean_qu', 'delta'].
    delta_data_emp : pandas.DataFrame
        Empirical delta data with at least the columns:
        ['quantile', 'mean_qu', 'delta'].
    caf_data : pandas.DataFrame
        Model CAF data with columns including:
        ['rt_bin', 'accuracy', <congruency>].
    caf_data_emp : pandas.DataFrame
        Empirical CAF data with columns including:
        ['rt_bin', 'accuracy', <congruency_emp>].
    cdf_data : pandas.DataFrame
        Long-format model RT data with columns:
        ['quantile', 'condition', 'rt'].
    cdf_data_emp : pandas.DataFrame
        Long-format empirical RT data with columns:
        ['quantile', 'condition', 'rt'].
    congruency : str, optional
        Column name in `caf_data` indicating congruency condition
        for the model (default: 'congruency').
    congruency_emp : str, optional
        Column name in `caf_data_emp` indicating congruency condition
        for the empirical data (default: 'congruency').
    n_delta_bins : int, optional
        Number of bins used when discretizing `mean_qu` with `pd.cut`
        (default: 10). Currently used when computing intermediate
        delta summaries.
    delta_ylim : tuple of float, optional
        Y-axis limits for the Δ-function subplot.
    delta_xlim : tuple of float, optional
        X-axis limits for the Δ-function subplot.
    cdf_xlim : tuple of float, optional
        X-axis limits for the CDF-function subplot.
    fontsize : int, optional
        Font size for subplot titles (default: 14).
    fontsize_axes : int, optional
        Font size for axis labels (default: 14).
    fontsize_ticklabels : int, optional
        Font size for tick labels (default: 10).
    fontsize_legend : int, optional
        Font size for the legend (default: 12).
    legend : bool, optional
        If True, draw a legend for the CAF panel (default: True).
    new_plot : bool, optional
        If True, create a new figure and axes. If False, draw into the
        provided `fig` and `axes` (default: True).
    caf_errorbars : object, optional
        Errorbar specification passed through to `sns.lineplot` for the model CAF.
        This can be any format accepted by seaborn's `errorbar` parameter
        (e.g. 'ci', 'se', None, a tuple, or a callable; default: None).
    hue_order : sequence of str, optional
        Order of condition levels for hue mapping (default: ('congruent', 'incongruent')).
    palette_emp : Mapping[str, str], optional
        Color palette for empirical lines, mapping condition names to hex colors
        (default: {"congruent": "#132a70", "incongruent": "#FF6361"}).
    palette_model : Mapping[str, str], optional
        Color palette for model lines, mapping condition names to hex colors
        (default: {"congruent": "#132a70", "incongruent": "#FF6361"}).
    delta_linestyle_model : str, optional
        Matplotlib linestyle for the model Δ-function line (default: '-').
    caf_linestyle_model : str, optional
        Matplotlib linestyle for the model CAF line (default: '-').
    cdf_linestyle_model : str, optional
        Matplotlib linestyle for the model CDF line (default: '-').
    linewidth : float, optional
        Line width for all plotted lines (default: 0.5).
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into when `new_plot` is False.
        Ignored if `new_plot` is True.
    axes : sequence of matplotlib.axes.Axes, optional
        Existing axes (length 3) to draw into when `new_plot` is False.
        Ignored if `new_plot` is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the 1×3 subplots.
    axes : sequence of matplotlib.axes.Axes
        The three axes objects for CAF, CDF, and Δ-function, respectively.
    """

    mean_data = cdf_data.groupby(['quantile', congruency])['rt'].mean().reset_index()

    mean_data_emp = cdf_data_emp.groupby(['quantile', congruency_emp])['rt'].mean().reset_index()

    if new_plot:
        fig, axes = plt.subplots(1,3, figsize=(12,3))

    # CAFs
    sns.lineplot(caf_data, 
                 linewidth=linewidth,
                 x='rt_bin', 
                 y='accuracy', 
                 hue=congruency, 
                 errorbar=caf_errorbars, 
                 ax=axes[0],
                 legend=False, 
                 hue_order=hue_order, 
                 palette=palette_model,
                 linestyle=caf_linestyle_model)
    
    
    sns.lineplot(caf_data_emp, 
                 linestyle='--',
                 marker="o", 
                 errorbar=None, 
                 legend=legend, 
                 linewidth=linewidth,
                 x='rt_bin', 
                 y='accuracy', 
                 hue=congruency_emp, 
                 ax=axes[0], 
                 hue_order=hue_order, 
                 palette=palette_emp)
    
    axes[0].set(ylim=(0, 1))
    axes[0].set_title('CAF', fontsize=fontsize)
    
    axes[1].set_title('CDF', fontsize=fontsize)
    axes[1].set_ylabel('Cumulative Density', fontsize=fontsize_axes)
    axes[1].set_xlabel('RT[s]', fontsize=fontsize_axes)
    axes[0].set_ylabel('CAF', fontsize=fontsize_axes)
    axes[0].set_xlabel('Bins', fontsize=fontsize_axes)

    # CDFs
    sns.lineplot(mean_data, 
                 linewidth=linewidth, 
                 linestyle=cdf_linestyle_model, 
                 x='rt', 
                 y='quantile', 
                 hue=congruency, 
                 alpha=1, 
                 ax=axes[1], 
                 legend=False, 
                 hue_order=hue_order, 
                 palette=palette_model)
    
    sns.lineplot(mean_data_emp, 
                 linewidth=linewidth, 
                 marker="o", 
                 linestyle='--', 
                 x='rt', 
                 y='quantile',
                 legend=False, 
                 hue=congruency_emp, 
                 alpha=1, 
                 ax=axes[1], 
                 hue_order=hue_order, 
                 palette=palette_emp)
    
    axes[1].set_title('CDF', fontsize=fontsize)
    axes[1].set_ylabel('Cumulative Density', fontsize=fontsize_axes)
    axes[1].set_xlabel('RT[s]', fontsize=fontsize_axes)

    # DELTA
    delta_data['mean_qu_bins'] = pd.cut(delta_data["mean_qu"], bins=n_delta_bins)
    delta_bins = delta_data.groupby('mean_qu_bins', observed=False)['delta'].mean().reset_index()
    delta_bins['bin_mid'] = delta_bins['mean_qu_bins'].apply(lambda x: x.mid)


    delta_bins = (
            delta_data
            .groupby('quantile', observed=False)[['mean_qu', 'delta']]
            .mean()
            .reset_index()
            .sort_values('mean_qu')
        )

    delta_data_emp['mean_qu_bins'] = pd.cut(delta_data_emp["mean_qu"], bins=n_delta_bins)
    delta_bins_emp = delta_data_emp.groupby('mean_qu_bins', observed=False)['delta'].mean().reset_index()
    delta_bins_emp['bin_mid'] = delta_bins_emp['mean_qu_bins'].apply(lambda x: x.mid)

    delta_bins_emp = (
            delta_data_emp
            .groupby('quantile', observed=False)[['mean_qu', 'delta']]
                .agg(
                    mean_qu=('mean_qu', 'mean'),
                    delta=('delta', 'mean'),
                    sd_delta=('delta', 'std')
                    )
            .reset_index()
            .sort_values('mean_qu')
        )

    sns.lineplot(delta_bins,linewidth=linewidth, linestyle=delta_linestyle_model, x='mean_qu', y='delta', legend=False, ax=axes[2], color='black')
    sns.lineplot(delta_bins_emp,linewidth=linewidth,linestyle='--',marker="o",  x='mean_qu', y='delta', legend=False, ax=axes[2], color='black')
    

    if plot_individual_deltas:
        # single Deltas
        sns.lineplot(
            delta_data,
            linewidth=0.5,
            #linestyle="--",
            #marker="o",
            x="mean_qu",
            y="delta",
            hue=id_individual,
            legend=False,
            ax=axes[2],
            alpha=alpha,
        )


    axes[2].set_ylabel(r'$\Delta$', fontsize=fontsize_axes)
    axes[2].set_xlabel('RT[s]', fontsize=fontsize_axes)
    axes[2].set_title(r'$\Delta$-Function', fontsize=fontsize)
    
    if cdf_xlim is not None:
        axes[2].set(xlim=cdf_xlim)

    if delta_ylim is not None:
        axes[2].set(ylim=delta_ylim)

    if delta_xlim is not None:
        axes[2].set(ylim=delta_ylim)

    if legend:
        axes[0].legend(title='', loc='lower right', fontsize=fontsize_legend, frameon=False)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=fontsize_ticklabels)  
        ax.tick_params(axis='y', labelsize=fontsize_ticklabels)  

    fig.tight_layout()

    return fig, axes



def plot_fit_ppc(
    caf_data: pd.DataFrame,
    cdf_data: pd.DataFrame,
    delta_data: pd.DataFrame,
    caf_data_emp: pd.DataFrame,
    cdf_data_emp: pd.DataFrame,
    delta_data_emp: pd.DataFrame,
    show_draws_caf: bool = True,
    caf_draws_errorbar: Optional[Tuple[str, float]] = None,
    show_draws_cdf: bool = True,
    show_draws_delta: bool = True,
    show_draws_mean: bool = False,
    draw_name: str = 'num_resim',
    congruency: str = "congruency",
    congruency_emp: str = "congruency",
    n_delta_bins: int = 10,
    delta_ylim: Optional[Tuple[float, float]] = None,
    delta_xlim: Optional[Tuple[float, float]] = None,
    cdf_xlim: Optional[Tuple[float, float]] = None,
    caf_ylim: Optional[Tuple[float, float]] = None,
    fontsize: int = 14,
    fontsize_axes: int = 16,
    fontsize_ticklabels: int = 10,
    fontsize_legend: int = 12,
    legend: bool = True,
    new_plot: bool = True,
    hue_order: Sequence[str] = ("congruent", "incongruent"),
    palette_emp: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    palette_model: Mapping[str, str] = {"congruent": "#132a70", "incongruent": "#FF6361"},
    delta_linestyle_model: str = "-",
    caf_linestyle_model: str = "-",
    cdf_linestyle_model: str = "-",
    empirical_marker: str = 'o',
    empirical_linestyle: str = '--',
    draw_linewidth: float = 0.5,
    empirical_linewidth: float = 0.5,
    fig: Optional[Figure] = None,
    axes: Optional[Sequence[Axes]] = None,
    draw_alpha: float = 0.05,
    mean_linewidth: float = 1,
    markeredgecolor: str ="none",
    markersize: float = 5):
    """
    Plot posterior predictive checks (PPC) for CAF, CDF-style quantile curves,
    and Δ-functions in a 1×3 subplot layout.

    This function compares model-generated summaries (e.g., posterior predictive
    draws or simulation-based summaries) against empirical data. It visualizes:

    1. **CAF (Conditional Accuracy Function)**:
    Accuracy as a function of RT bins, separated by congruency.

    2. **CDF-style quantile plot**:
    Vincentized RT quantiles (mean RT per quantile) as a function of cumulative
    probability, separated by congruency.

    3. **Δ-function**:
    Difference between incongruent and congruent RT quantiles as a function of
    the mean RT at each quantile.

    Model predictions can be displayed as:
    - **draw-wise curves**: individual posterior predictive draws shown as
    semi-transparent lines
    - **aggregated mean curves**: averages across draws

    Empirical data are overlaid as stylized line-and-marker plots.

    Parameters
    ----------
    caf_data : pandas.DataFrame
        Model CAF data. Expected columns include:

        - ``"rt_bin"``
        - ``"accuracy"``
        - a congruency column specified by ``congruency``
        - a draw identifier column specified by ``draw_name`` when
        ``show_draws_caf=True``

    cdf_data : pandas.DataFrame
        Model CDF-style quantile data in long format. Expected columns include:

        - ``"quantile"``
        - ``"rt"``
        - a congruency column specified by ``congruency``
        - a draw identifier column specified by ``draw_name`` when
        ``show_draws_cdf=True``

    delta_data : pandas.DataFrame
        Model delta-function data. Expected columns include:

        - ``"quantile"``
        - ``"mean_qu"``
        - ``"delta"``
        - a draw identifier column specified by ``draw_name`` when
        ``show_draws_delta=True``

    caf_data_emp : pandas.DataFrame
        Empirical CAF data. Expected columns include:

        - ``"rt_bin"``
        - ``"accuracy"``
        - a congruency column specified by ``congruency_emp``

    cdf_data_emp : pandas.DataFrame
        Empirical CDF-style quantile data. Expected columns include:

        - ``"quantile"``
        - ``"rt"``
        - a congruency column specified by ``congruency_emp``

    delta_data_emp : pandas.DataFrame
        Empirical delta-function data. Expected columns include:

        - ``"quantile"``
        - ``"mean_qu"``
        - ``"delta"``

    show_draws_caf : bool, optional
        If True, plot individual posterior predictive draws for the CAF panel as
        semi-transparent lines. Each line corresponds to one simulated draw.
        Default is True.

    ccaf_draws_errorbar : tuple | str | callable, optional
        Error bar specification passed to ``seaborn.lineplot`` for the CAF panel
        when plotting aggregated model predictions (i.e., when ``include_mean=True``).

        This follows seaborn’s ``errorbar`` API and can be:

        - ``("pi", 95)`` for a 95% percentile interval (default choice for PPC-style uncertainty)
        - ``"ci"`` or ``("ci", level)`` for confidence intervals
        - ``"se"`` or ``"sd"`` for standard error or standard deviation

    show_draws_cdf : bool, optional
        If True, plot individual posterior predictive draws for the CDF-style
        quantile panel as semi-transparent lines. Each line corresponds to one
        simulated draw. Default is True.

    show_draws_delta : bool, optional
        If True, plot individual posterior predictive draws for the Δ-function
        panel as semi-transparent lines. Each line corresponds to one simulated
        draw. Default is True.

    show_draws_mean : bool, optional
        If True, overlay mean model predictions aggregated across draws
        (default: False).

    draw_name : str, optional
        Column name identifying posterior predictive draws or resimulations
        (default: ``"num_resim"``).

    congruency : str, optional
        Column name for congruency in model data (default: ``"congruency"``).

    congruency_emp : str, optional
        Column name for congruency in empirical data (default: ``"congruency"``).

    n_delta_bins : int, optional
        Number of bins for intermediate delta summaries. Currently retained for
        compatibility; the plotted delta summaries are aggregated by quantile.

    delta_ylim : tuple of float, optional
        Y-axis limits for the Δ-function subplot.

    delta_xlim : tuple of float, optional
        X-axis limits for the Δ-function subplot.

    cdf_xlim : tuple of float, optional
        X-axis limits for the CDF subplot.

    caf_ylim : tuple of float, optional
        Y-axis limits for the CAF subplot.

    fontsize : int, optional
        Font size for subplot titles (default: 14).

    fontsize_axes : int, optional
        Font size for axis labels (default: 14).

    fontsize_ticklabels : int, optional
        Font size for tick labels (default: 10).

    fontsize_legend : int, optional
        Font size for the legend (default: 12).

    legend : bool, optional
        Whether to display a legend (default: True).

    new_plot : bool, optional
        If True, create a new figure. Otherwise, draw into the provided ``fig`` and
        ``axes`` (default: True).

    hue_order : sequence of str, optional
        Order of condition levels for plotting
        (default: ``("congruent", "incongruent")``).

    palette_emp : Mapping[str, str], optional
        Color mapping for empirical conditions.

    palette_model : Mapping[str, str], optional
        Color mapping for model predictions.

    delta_linestyle_model, caf_linestyle_model, cdf_linestyle_model : str, optional
        Line styles for model predictions in the respective panels.

    empirical_marker : str, optional
        Marker style for empirical data (default: ``"o"``).

    empirical_linestyle : str, optional
        Line style for empirical data (default: ``"--"``).

    draw_linewidth : float, optional
        Line width for individual draw-wise model curves.

    empirical_linewidth : float, optional
        Line width for empirical curves.

    fig : matplotlib.figure.Figure, optional
        Existing figure for plotting if ``new_plot`` is False.

    axes : sequence of matplotlib.axes.Axes, optional
        Existing axes (length 3) for plotting if ``new_plot`` is False.

    draw_alpha : float, optional
        Transparency for individual draw-wise model curves (default: 0.05).

    mean_linewidth : float, optional
        Line width for aggregated model curves.

    markeredgecolor : str, optional
        Edge color for empirical markers (default: ``"none"``).

    markersize : float, optional
        Size of empirical markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.

    axes : sequence of matplotlib.axes.Axes
        Axes corresponding to CAF, CDF-style quantile plot, and Δ-function.

    Examples
    --------
    >>> fig, axes = plot_fit_ppc(
    ...     caf_data, cdf_data, delta_data,
    ...     caf_emp, cdf_emp, delta_emp
    ... )

    >>> fig, axes = plot_fit_ppc(
    ...     caf_data, cdf_data, delta_data,
    ...     caf_emp, cdf_emp, delta_emp,
    ...     show_draws_caf=True,
    ...     show_draws_cdf=True,
    ...     show_draws_delta=True,
    ...     show_draws_mean=True,
    ...     draw_alpha=0.05
    ... )
    """

    if new_plot:
        fig, axes = plt.subplots(1,3, figsize=(16,4))

    # CAFs

    # Aggregated Prediction
    if show_draws_mean:
        sns.lineplot(caf_data, 
                linewidth=mean_linewidth,
                x='rt_bin', 
                y='accuracy', 
                hue=congruency, 
                errorbar=caf_draws_errorbar,
                err_kws={"alpha": 0.08}, 
                alpha=1,
                ax=axes[0],
                legend=False, 
                hue_order=hue_order, 
                palette=palette_model,
                linestyle=caf_linestyle_model)

    # Spaghetti Predictions
    if show_draws_caf:
        sns.lineplot(caf_data, 
                    linewidth=draw_linewidth,
                    x='rt_bin', 
                    y='accuracy', 
                    hue=congruency, 
                    errorbar=None, 
                    estimator=None,
                    units=draw_name,
                    alpha=draw_alpha,
                    ax=axes[0],
                    legend=False, 
                    hue_order=hue_order, 
                    palette=palette_model,
                    linestyle=caf_linestyle_model)
    
    # Empirical Data
    sns.lineplot(caf_data_emp, 
                 linestyle=empirical_linestyle,
                 linewidth=empirical_linewidth,
                 marker=empirical_marker, 
                 errorbar=None, 
                 legend=legend, 
                 markeredgecolor=markeredgecolor,
                 markersize=markersize,
                 x='rt_bin', 
                 y='accuracy', 
                 hue=congruency_emp, 
                 ax=axes[0], 
                 hue_order=hue_order, 
                 palette=palette_emp)

    
    axes[0].set(ylim=(0, 1))
    axes[0].set_title('CAF', fontsize=fontsize)
    axes[0].set_ylabel('CAF', fontsize=fontsize_axes)
    axes[0].set_xlabel('Bins', fontsize=fontsize_axes)

    # CDFs

    mean_data = cdf_data.groupby(['quantile', congruency])['rt'].mean().reset_index()

    mean_data_emp = cdf_data_emp.groupby(['quantile', congruency_emp])['rt'].mean().reset_index()

    # Spaghetti Predictions
    if show_draws_cdf:
        sns.lineplot(cdf_data, 
                    linewidth=draw_linewidth, 
                    linestyle=cdf_linestyle_model, 
                    x='rt', 
                    y='quantile', 
                    estimator=None,
                    units=draw_name,
                    hue=congruency, 
                    alpha=draw_alpha, 
                    ax=axes[1], 
                    legend=False, 
                    hue_order=hue_order, 
                    palette=palette_model)
    
    # Aggregated Predictions
    if show_draws_mean:
        sns.lineplot(mean_data, 
                linewidth=mean_linewidth, 
                linestyle=cdf_linestyle_model, 
                x='rt', 
                y='quantile', 
                hue=congruency, 
                alpha=1, 
                ax=axes[1], 
                legend=False, 
                hue_order=hue_order, 
                palette=palette_model)
    
    # Empirical Data
    sns.lineplot(mean_data_emp, 
                 linewidth=empirical_linewidth, 
                 marker=empirical_marker, 
                 linestyle=empirical_linestyle, 
                 x='rt', 
                 y='quantile',
                 legend=False,
                 markeredgecolor=markeredgecolor,
                 markersize=markersize,
                 hue=congruency_emp, 
                 alpha=1, 
                 ax=axes[1], 
                 hue_order=hue_order, 
                 palette=palette_emp)
    
    axes[1].set_title('CDF', fontsize=fontsize)
    axes[1].set_ylabel('Cumulative Density', fontsize=fontsize_axes)
    axes[1].set_xlabel('RT[s]', fontsize=fontsize_axes)

    # DELTA
    delta_data['mean_qu_bins'] = pd.cut(delta_data["mean_qu"], bins=n_delta_bins)

    delta_bins = (
            delta_data
            .groupby('quantile', observed=False)[['mean_qu', 'delta']]
                .agg(
                    mean_qu=('mean_qu', 'mean'),
                    delta=('delta', 'mean'),
                    sd_delta=('delta', 'std')
                    )
            .reset_index()
            .sort_values('mean_qu')
        )

    delta_data_emp['mean_qu_bins'] = pd.cut(delta_data_emp["mean_qu"], bins=n_delta_bins)

    delta_bins_emp = (
            delta_data_emp
            .groupby('quantile', observed=False)[['mean_qu', 'delta']]
                .agg(
                    mean_qu=('mean_qu', 'mean'),
                    delta=('delta', 'mean'),
                    sd_delta=('delta', 'std')
                    )
            .reset_index()
            .sort_values('mean_qu')
        )

    # Spaghetti Predictions
    if show_draws_delta:
        sns.lineplot(delta_data,
                    linewidth=draw_linewidth, 
                    linestyle=delta_linestyle_model, 
                    x='mean_qu', 
                    y='delta', 
                    estimator=None,
                    units=draw_name,
                    legend=False, 
                    ax=axes[2], 
                    alpha = draw_alpha,
                    color='black')
    
    # Aggregated Predictions
    if show_draws_mean:
        sns.lineplot(delta_bins,
                     linewidth=mean_linewidth,
                     linestyle=delta_linestyle_model,
                     x='mean_qu', 
                     y='delta', 
                     alpha=0.8,
                     legend=False,
                     color='#0A2A5E',
                     ax=axes[2])

    # Empirical Data
    sns.lineplot(delta_bins_emp,
                 linewidth=empirical_linewidth,
                 linestyle=empirical_linestyle,
                 marker=empirical_marker,
                 markeredgecolor=markeredgecolor,
                 markersize=markersize,
                 x='mean_qu', 
                 y='delta', 
                 legend=False, 
                 ax=axes[2],
                 color='black')

    axes[2].set_ylabel(r'$\Delta$', fontsize=fontsize_axes)
    axes[2].set_xlabel('RT[s]', fontsize=fontsize_axes)
    axes[2].set_title(r'$\Delta$-Function', fontsize=fontsize)

    if cdf_xlim is not None:
        axes[1].set(xlim=cdf_xlim)
    
    if caf_ylim is not None:
        axes[0].set(ylim=caf_ylim)

    if delta_ylim is not None:
        axes[2].set(ylim=delta_ylim)

    if delta_xlim is not None:
        axes[2].set(xlim=delta_xlim)

    if legend & new_plot:
        axes[0].legend(title='', loc='lower right', fontsize=fontsize_legend, frameon=False)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=fontsize_ticklabels)  
        ax.tick_params(axis='y', labelsize=fontsize_ticklabels)  

    fig.tight_layout()

    return fig, axes

def summarise_q_ppc(
    data: pd.DataFrame,
    grouping_vars: List[str] = None,
    id_name: str = "id",
    rt: str = "rt",
    accuracy: str = "accuracy",
    congruency: str = "congruency",
) -> pd.DataFrame:
    """
    Compute grouped RT quantiles, mean RT, and mean accuracy for posterior
    predictive checks or descriptive model-fit summaries.

    The function aggregates trial-level data within the groups defined by
    `grouping_vars` and returns a wide-format summary table containing:

    - RT quantiles (25th, 50th, and 75th percentiles)
    - mean RT
    - mean accuracy

    RT quantiles and mean RT are computed within the full grouping structure
    given by `grouping_vars`. If `grouping_vars` include `accuracy`, Mean accuracy is computed after removing the
    `accuracy column from the grouping variables, if present, so that accuracy
    is summarized across trials rather than within a fixed accuracy category.

    Parameters
    ----------
    data : pandas.DataFrame
        Trial-level data containing at least the columns specified by
        `id_name`, `rt`, `accuracy`, and `congruency`.

    grouping_vars : list of str
        Column names defining the grouping structure for the summaries.
        Examples are `["id", "congruency"]` or
        `["id", "congruency", "accuracy"]`.

        If `accuracy` is included in `grouping_vars`, RT quantiles and mean RT
        are computed separately for each accuracy level, but mean accuracy is
        still computed after removing `accuracy` from the grouping variables.

    id_name : str, default="id"
        Column name identifying subjects or independent observational units.

    rt : str, default="rt"
        Column name containing reaction times.

    accuracy : str, default="accuracy"
        Column name containing response accuracy, typically coded as 0/1.

    congruency : str, default="congruency"
        Column name containing congruency-condition labels.

    Returns
    -------
    pandas.DataFrame
        A wide-format DataFrame with one row per grouping cell and the
        following columns:

        - grouping variables from `grouping_vars`
        - `mean_rt`: mean RT within group
        - `mean_acc`: mean accuracy within group
        - `rt_q25`: 25th RT percentile within group
        - `rt_q50`: 50th RT percentile (median) within group
        - `rt_q75`: 75th RT percentile within group


    Raises
    ------
    KeyError
        If one or more required columns are missing from `data`.

    Notes
    -----
    - `dmc_helpers.check_vars()` is used to validate that the required columns
      are present.
    - `dmc_helpers.check_congruency()` standardizes congruency labels to
      `"congruent"` and `"incongruent"` before aggregation.
    - RT quantiles are computed over all rows within each grouping cell.
      Including `accuracy` in `grouping_vars` creates separate RT summaries for
      each accuracy level; it does not filter the data to correct trials only.
    - The function returns a summary table intended for PPC visualization or
      empirical-versus-simulated fit comparisons.
    """

    # check if variables are present in data set
    check_vars(data, rt=rt, id_name=id_name, accuracy=accuracy, congruency=congruency)

    # check congruency coding and return data with 'congruent'/'incongruent' labels
    data = check_congruency(
        data,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    grouping_vars_acc = grouping_vars.copy()

    if accuracy in grouping_vars_acc:
        grouping_vars_acc.remove(accuracy)

    # compute RT quantiles for each condition
    df_q = (
        data
        .groupby(grouping_vars)[rt]
        .quantile([0.25, 0.5, 0.75])
        .rename_axis(index=[*grouping_vars, "quantile"])
        .reset_index()
    )

    # compute mean RTs for each condition
    df_means_rt = (
        data
        .groupby(grouping_vars)
        .agg(
            mean_rt=(rt, "mean"),
        )
        .reset_index()
    )

    # compute mean Accuracies for each condition 
    # (of course not per accuracy condition)
    df_means_acc = (
        data
        .groupby(grouping_vars_acc)
        .agg(
            mean_acc=(accuracy, "mean"),
        )
        .reset_index()
    )

    # merge all data sets
    df_q = (df_q
        .merge(df_means_rt, on=grouping_vars)
        .merge(df_means_acc, on=grouping_vars_acc)
        )

    # transform to wide and rename columns
    df_q_wide = (
        df_q
        .pivot_table(
            index=grouping_vars + ["mean_rt", "mean_acc"],
            columns="quantile",
            values=rt,
        )
        .reset_index()
    )

    df_q_wide = df_q_wide.rename(
        columns={
            0.25: "rt_q25",
            0.50: "rt_q50",
            0.75: "rt_q75",
        }
    )

    # make sure accuracy is an integer
    if accuracy in grouping_vars:
        df_q_wide[accuracy] = df_q_wide[accuracy].astype(int)

    return df_q_wide


def compute_fit_qs(
    resimulated_data: pd.DataFrame,
    empirical_data: pd.DataFrame,
    grouping_vars: List[str],
    draw_name: str = None,
    summarise_draws: bool = True,
    id_name: str = 'id',
    rt: str = "rt",
    accuracy: str = "accuracy",
    congruency: str = "congruency"
) -> pd.DataFrame:
    
    """
    Compute grouped empirical and posterior-predictive summary statistics and
    merge them into a single comparison table.

    For both empirical and resimulated trial-level data, the function computes
    distributional summaries using `summarise_q_ppc()`. These summaries include:

    - mean RT
    - mean accuracy
    - RT quantiles (25th, 50th, 75th percentiles)

    Empirical summaries are always computed at the grouping level specified by
    `grouping_vars`.

    If `draw_name` is provided, resimulated summaries are first computed within
    each draw as well as within `grouping_vars`. If `summarise_draws=True`, these
    draw-level summaries are then aggregated across draws within each grouping
    cell using:

    - the median
    - the 5th percentile
    - the 95th percentile

    This yields one posterior-predictive point estimate and interval per grouping
    cell, which can be merged with the corresponding empirical summary.

    Parameters
    ----------
    resimulated_data : pandas.DataFrame
        Trial-level model-generated data. If `draw_name` is provided, this
        DataFrame must contain a column identifying resimulation draws.

    empirical_data : pandas.DataFrame
        Trial-level empirical data.

    grouping_vars : list of str
        Column names defining the grouping structure for empirical summaries and
        for the final merge. Examples include `["id", "congruency"]` or
        `["id", "congruency", "accuracy"]`.

    draw_name : str or None, default=None
        Column name identifying posterior-predictive draws or resimulation
        indices in `resimulated_data`. If provided, resimulated summaries are
        computed separately for each draw.

    summarise_draws : bool, default=True
        If True and `draw_name` is provided, aggregate the draw-level simulated
        summaries across draws within each grouping cell using the median and
        the 5th and 95th percentiles. If False, keep draw-level summaries.

    id_name : str, default='id'
        Column name identifying participants or observational units.

    rt : str, default='rt'
        Column name containing reaction times.

    accuracy : str, default='accuracy'
        Column name containing response accuracy, typically coded as 0/1.

    congruency : str, default='congruency'
        Column name containing congruency-condition labels.

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame containing empirical and simulated summary
        statistics.

        If `summarise_draws=True`, the output contains one row per grouping cell
        in `grouping_vars`, with empirical columns suffixed by `_emp` and
        simulated columns named with suffix patterns such as:

        - `_resim_median`
        - `_resim_q05`
        - `_resim_q95`

        If `summarise_draws=False`, the output contains draw-level simulated
        summaries merged with the corresponding empirical summaries.

    Raises
    ------
    ValueError
        If `draw_name` is provided but is not present in `resimulated_data`.

    Notes
    -----
    - This function prepares grouped summaries for posterior predictive checks
    and descriptive fit assessment; it does not compute formal fit metrics.
    - Empirical and simulated summaries are generated by `summarise_q_ppc()`.
    - Percentile-based intervals are centered naturally around the simulated
    median rather than the simulated mean.
    """
    grouping_vars = list(grouping_vars)

    # compute summary stats for empirical data
    df_q_emp_wide = summarise_q_ppc(
        empirical_data,
        rt=rt,
        id_name=id_name,
        accuracy=accuracy,
        congruency=congruency,
        grouping_vars=grouping_vars,
    )

    if draw_name is not None:
        if draw_name not in resimulated_data.columns:
            raise ValueError(
                f"draw_name '{draw_name}' not present in resimulated_data. "
                "Please provide a valid resimulation index column."
            )

        grouping_vars_resim = grouping_vars.copy()
        if draw_name not in grouping_vars_resim:
            grouping_vars_resim.append(draw_name)
    else:
        grouping_vars_resim = grouping_vars.copy()
        summarise_draws = False

    # compute summary stats for resimulated data
    df_q_wide = summarise_q_ppc(
        resimulated_data,
        rt=rt,
        id_name=id_name,
        accuracy=accuracy,
        congruency=congruency,
        grouping_vars=grouping_vars_resim,
    )

    if summarise_draws:
        var_names = ["mean_rt", "mean_acc", "rt_q25", "rt_q50", "rt_q75"]

        # append 'resim' to column names
        cols_to_rename = [c for c in df_q_wide.columns if c not in grouping_vars]
        df_q_wide = df_q_wide.rename(columns={c: f"{c}_resim" for c in cols_to_rename})
        new_vars = [f'{c}_resim' for c in var_names]

        # compute summary statistics across draws
        df_q_wide = (
            df_q_wide.groupby(grouping_vars, observed=False)[new_vars]
            .agg([
                "median",
                ("q05", lambda x: np.quantile(x, 0.05)),
                ("q95", lambda x: np.quantile(x, 0.95)),
            ])
            .reset_index()
        )

        # reduce levels of df
        df_q_wide.columns = [
            col if not isinstance(col, tuple) else "_".join([str(c) for c in col if c != ""])
            for col in df_q_wide.columns
        ]

        # append '_emp' to empirical data
        cols_to_rename = [c for c in df_q_emp_wide.columns if c not in grouping_vars]
        df_q_emp_wide = df_q_emp_wide.rename(columns={c: f"{c}_emp" for c in cols_to_rename})

    data_merged = pd.merge(
        df_q_wide,
        df_q_emp_wide,
        how="left",
        on=grouping_vars,
        suffixes=("_resim", "_emp")
    )

    return data_merged

def plot_fit_qs(
    data: pd.DataFrame,
    con_color: str = "#10225e",
    inc_color: str = "#FF6361",
    fontsize: int = 22,
    accuracy_lims: Tuple[float, float] = (0.6, 1.0),
    figsize: Tuple[float, float] = (15, 3),
    plot_uncertainty: bool = False,
    **kwargs: Any
) -> Tuple[Figure, list[Axes]]:
    """
    Visualize quantile-based model fit by comparing empirical and resimulated
    summary statistics.

    This function creates a five-panel scatterplot comparing empirical summary
    statistics to corresponding resimulated or posterior-predictive summaries.
    Each panel plots empirical values on the x-axis against simulated values on
    the y-axis, with a dashed diagonal reference line (`y = x`) indicating
    perfect agreement.

    The following statistics are shown:

    - mean RT
    - mean accuracy
    - 25th RT percentile
    - 50th RT percentile (median)
    - 75th RT percentile

    Points are colored by congruency condition, with expected levels
    `"congruent"` and `"incongruent"`.

    Two input modes are supported:

    1. **Point-estimate mode** (`plot_uncertainty=False`)
    The input `data` must contain empirical columns with suffix `_emp` and
    simulated columns with suffix `_resim`, for example:

    - `mean_rt_emp`, `mean_rt_resim`
    - `mean_acc_emp`, `mean_acc_resim`
    - `rt_q25_emp`, `rt_q25_resim`
    - `rt_q50_emp`, `rt_q50_resim`
    - `rt_q75_emp`, `rt_q75_resim`

    2. **Uncertainty mode** (`plot_uncertainty=True`)
    The input `data` must contain empirical columns with suffix `_emp` and
    aggregated posterior-predictive summaries with suffixes `_resim_median`,
    `_resim_q05`, and `_resim_q95` auch as computed by compute_fit_qs(), for example:

    - `mean_rt_emp`, `mean_rt_resim_median`, `mean_rt_resim_q05`, `mean_rt_resim_q95`
    - `mean_acc_emp`, `mean_acc_resim_median`, `mean_acc_resim_q05`, `mean_acc_resim_q95`
    - `rt_q25_emp`, `rt_q25_resim_median`, `rt_q25_resim_q05`, `rt_q25_resim_q95`
    - `rt_q50_emp`, `rt_q50_resim_median`, `rt_q50_resim_q05`, `rt_q50_resim_q95`
    - `rt_q75_emp`, `rt_q75_resim_median`, `rt_q75_resim_q05`, `rt_q75_resim_q95`

    In this mode, vertical error bars show the interval from the 5th to the
    95th percentile of the simulated summaries.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing merged empirical and simulated summary statistics.

    con_color : str, default="#10225e"
        Color used for the `"congruent"` condition.

    inc_color : str, default="#FF6361"
        Color used for the `"incongruent"` condition.

    fontsize : int, default=22
        Base font size used for subplot titles and shared axis labels.

    accuracy_lims : tuple of float, default=(0.6, 1.0)
        Axis limits for the mean-accuracy panel. The same limits are applied to
        both x- and y-axes.

    figsize : tuple of float, default=(15, 3)
        Figure size in inches.

    plot_uncertainty : bool, default=False
        If False, plot only point estimates from columns ending in `_resim`.
        If True, plot simulated medians from columns ending in `_resim_median`
        and add vertical uncertainty bars using the corresponding `_resim_q05`
        and `_resim_q95` columns.

    **kwargs : Any
        Additional keyword arguments passed to `seaborn.scatterplot`. These can
        be used to override default marker aesthetics such as size, alpha, or
        linewidth.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.

    axes : list of matplotlib.axes.Axes
        List of the five subplot axes in left-to-right order.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain the columns required for the
        requested plotting mode.

    Notes
    -----
    - The function assumes that congruency labels have already been standardized
    to `"congruent"` and `"incongruent"`.
    - In uncertainty mode, the simulated median is plotted as the point estimate
    and the 5th and 95th percentiles are plotted as vertical uncertainty bars.
    - Axis limits for RT panels are computed from the combined range of empirical
    and simulated values, and also include uncertainty bounds when
    `plot_uncertainty=True`.
    """

    hue_order = ["congruent", "incongruent"]
    palette = {"congruent": con_color, "incongruent": inc_color}

    titles = ["Mean RT", "Mean Accuracy", "25% Quantile RT", "Median RT", "75% Quantile RT"]
    stats = ["mean_rt", "mean_acc", "rt_q25", "rt_q50", "rt_q75"]

    plot_data = data.copy()

    if plot_uncertainty:
        resim_suffix = 'resim_median'

        expected_unc_cols = (
            [f"{v}_resim_median" for v in stats]
            + [f"{v}_resim_q05" for v in stats]
            + [f"{v}_resim_q95" for v in stats]
        )

        missing_unc_cols = [c for c in expected_unc_cols if c not in plot_data.columns]

        if missing_unc_cols:
            raise ValueError(
                "data does not include the columns required for uncertainty plotting. "
                "Please set plot_uncertainty=False or summarise_draws=True in compute_fit_qs(). "
                f"Missing columns: {missing_unc_cols}"
            )

    else:
        resim_suffix = 'resim'

        if 'mean_rt_resim_median' in plot_data.columns:
            raise ValueError(
                "data includes aggregated values across resimulations. "
                "Please set plot_uncertainty=True."
            )

    fig, axes_arr = plt.subplots(1, 5, figsize=figsize)

    # ensure a stable return type
    axes = list(axes_arr)

    for j, var in enumerate(stats):
        x_col = f"{var}_emp"
        y_col = f"{var}_{resim_suffix}"

        scatter_kws = dict(
            data=plot_data,
            x=x_col,
            y=y_col,
            hue="congruency",
            hue_order=hue_order,
            palette=palette,
            alpha=0.5,
            s=12,
            linewidth=0,
            legend=False,
            marker="o",
            ax=axes[j],
        )
        scatter_kws.update(kwargs)
        sns.scatterplot(**scatter_kws)

        if plot_uncertainty:
            for level in hue_order:
                sub = plot_data[plot_data["congruency"] == level]

                axes[j].errorbar(
                    sub[x_col],
                    sub[y_col],
                    yerr=[
                        sub[y_col] - sub[f"{var}_resim_q05"],
                        sub[f"{var}_resim_q95"] - sub[y_col],
                    ],
                    fmt="none",
                    ecolor=palette[level],
                    elinewidth=0.5,
                    alpha=0.5
                )

        if var != "mean_acc":
            series_to_bound = [plot_data[x_col], plot_data[y_col]]
            if plot_uncertainty:
                series_to_bound.extend([
                    plot_data[f"{var}_resim_q05"],
                    plot_data[f"{var}_resim_q95"],
                ])

            all_vals = pd.concat(series_to_bound)
            vmin = all_vals.min(skipna=True)
            vmax = all_vals.max(skipna=True)

            if pd.isna(vmin) or pd.isna(vmax):
                lims = [0.0, 1.0]
            else:
                lims = [float(vmin) - 0.02, float(vmax) + 0.02]
        else:
            lims = list(accuracy_lims)

        axes[j].plot(lims, lims, color="black", linestyle="--", linewidth=1)
        axes[j].set_xlim(lims)
        axes[j].set_ylim(lims)

        axes[j].set_xlabel("")
        axes[j].set_ylabel("")
        axes[j].set_title(titles[j], fontsize=fontsize - 5)

    fig.supxlabel("Empirical", fontsize=fontsize - 5, y=0.0)
    fig.supylabel("Resimulated", fontsize=fontsize - 5, x=0.0)
    fig.tight_layout()

    return fig, axes

def summarise_q(
    data: pd.DataFrame,
    rt: str = "rt",
    accuracy: str = "accuracy",
    congruency: str = "congruency",
    grouping_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute quantile-based and mean summary statistics of response times (RT)
    and accuracy within groups.

    This function aggregates trial-level data into distributional summaries
    commonly used for model fit evaluation (e.g., quantile-based fit of RT
    distributions). For each group defined by `grouping_vars`, the function
    computes:

        - RT quantiles (25th, 50th, 75th percentiles)
        - Mean RT
        - Mean accuracy

    The resulting DataFrame is returned in wide format, with one row per group
    and separate columns for each RT quantile and mean statistic.

    Parameters
    ----------
    data
        Trial-level DataFrame containing at minimum RT, accuracy, and
        congruency columns.
    rt
        Column name containing response times (numeric).
    accuracy
        Column name containing response accuracy (numeric; typically 0/1).
    congruency
        Column name indicating congruency condition labels.
    grouping_vars
        Columns used to define grouping structure (e.g., participant and
        condition). If None, defaults to ["participant", "congruency"].

    Returns
    -------
    pandas.DataFrame
        A wide-format DataFrame with one row per group defined by
        `grouping_vars`. Contains:

            - Grouping variables
            - mean_rt : mean response time per group
            - mean_acc : mean accuracy per group
            - rt_q25 : 25th percentile of RT
            - rt_q50 : 50th percentile (median) of RT
            - rt_q75 : 75th percentile of RT

    Notes
    -----
    - `check_vars()` is used to validate the presence and format of required
      columns.
    - `check_congruency()` standardizes congruency coding before aggregation.
    - RT quantiles are computed across all trials within each group; if
      quantiles should be restricted (e.g., correct trials only), filtering
      must be applied prior to calling this function.
    """
    if grouping_vars is None:
        grouping_vars = ["participant", "congruency"]

    check_vars(data, rt=rt, accuracy=accuracy, congruency=congruency)

    data = check_congruency(
        data,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    df_q = (
        data
        .groupby(grouping_vars)[rt]
        .quantile([0.25, 0.5, 0.75])
        .rename_axis(index=[*grouping_vars, "quantile"])
        .reset_index()
    )

    df_means = (
        data
        .groupby(grouping_vars)
        .agg(
            mean_rt=(rt, "mean"),
            mean_acc=(accuracy, "mean"),
        )
        .reset_index()
    )

    df_q = df_q.merge(df_means, on=grouping_vars)

    df_q_wide = (
        df_q
        .pivot_table(
            index=grouping_vars + ["mean_rt", "mean_acc"],
            columns="quantile",
            values=rt,
        )
        .reset_index()
    )

    df_q_wide = df_q_wide.rename(
        columns={
            0.25: "rt_q25",
            0.50: "rt_q50",
            0.75: "rt_q75",
        }
    )

    return df_q_wide


def make_strictly_increasing(
    edges: Iterable[float],
    eps: float = 1e-9
) -> npt.NDArray[np.float64]:
    """
    Ensure a sequence of numeric values is strictly monotonically increasing.

    This function enforces strict monotonicity by scanning the input array
    from left to right and adjusting any non-increasing element so that it
    exceeds its predecessor by at least `eps`. The adjustment is performed
    in-place on a copied NumPy array, leaving the original input unchanged.

    This utility is particularly useful when constructing bin edges for
    histogramming or quantile-based discretization, where duplicate or
    non-increasing edges can cause downstream numerical or categorical
    binning errors.

    Parameters
    ----------
    edges : Iterable[float]
        A one-dimensional sequence of numeric values intended to represent
        ordered boundaries (e.g., histogram bin edges). The input does not
        need to be strictly increasing.
    eps : float, optional
        The minimum increment enforced between adjacent values when a
        violation of strict monotonicity is detected. Default is 1e-9.

    Returns
    -------
    numpy.ndarray
        A one-dimensional NumPy array of dtype float64 with strictly
        increasing values.

    Notes
    -----
    - The function guarantees `edges[i] > edges[i-1]` for all `i > 0`.
    - Adjustments are minimal and only applied when necessary.
    - The magnitude of `eps` should be chosen with respect to the numerical
      scale of `edges` to avoid unintended distortion.

    Examples
    --------
    >>> make_strictly_increasing([0.0, 1.0, 1.0, 2.0])
    array([0.0, 1.0, 1.000000001, 2.0])

    >>> make_strictly_increasing([3, 2, 1])
    array([3.0, 3.000000001, 3.000000002])
    """
    edges_array = np.asarray(edges, dtype=float).copy()

    for k in range(1, len(edges_array)):
        if edges_array[k] <= edges_array[k - 1]:
            edges_array[k] = edges_array[k - 1] + eps

    return edges_array


def get_bin_edges(
    rt: Iterable[float],
    quantiles: npt.ArrayLike = np.linspace(0.1, 0.9, 9),
) -> Optional[npt.NDArray[np.float64]]:
    """
    Construct strictly increasing quantile-based bin edges for response times.

    This function computes empirical quantiles of the provided response time
    (RT) sample and returns bin edges suitable for discretization (e.g.,
    histogramming or multinomial likelihood construction). The returned edges
    are bounded by negative and positive infinity to ensure full coverage of
    the support.

    Non-finite RT values (NaN, ±inf) are removed prior to quantile estimation.
    If no finite observations remain, the function returns None.

    To guard against numerical degeneracy (e.g., repeated quantiles due to ties),
    the resulting edges are passed through `make_strictly_increasing`, ensuring
    strict monotonicity.

    Parameters
    ----------
    rt : Iterable[float]
        One-dimensional sequence of response times. May contain non-finite
        values, which will be removed prior to quantile computation.
    quantiles : array-like, optional
        Sequence of quantile probabilities in the interval [0, 1] used to
        define internal bin boundaries. Default is nine equally spaced
        quantiles from 0.1 to 0.9 (inclusive), yielding ten bins.

    Returns
    -------
    numpy.ndarray or None
        A one-dimensional NumPy array of strictly increasing bin edges
        with the form:

            [-inf, q1, q2, ..., qK, +inf]

        where q1...qK are empirical quantiles of the finite RT values.

        Returns None if no finite RT observations are available.

    Notes
    -----
    - The number of resulting bins equals len(quantiles) + 1.
    - Quantile-based binning yields approximately equal expected counts
      per bin under the empirical distribution.
    - Strict monotonicity is enforced to prevent downstream errors in
      functions such as `pandas.cut` or histogram-based likelihood
      computations.
    - The function assumes `rt` represents a univariate distribution.

    Examples
    --------
    >>> rt = [0.35, 0.42, 0.51, 0.60, 0.72]
    >>> get_bin_edges(rt)
    array([-inf, 0.392, 0.434, ..., 0.688, inf])

    >>> get_bin_edges([np.nan, np.inf])
    None
    """
    rt_array = np.asarray(rt, dtype=float)
    rt_array = rt_array[np.isfinite(rt_array)]

    if rt_array.size == 0:
        return None

    q = np.quantile(rt_array, quantiles)
    edges = np.concatenate(([-np.inf], q, [np.inf]))

    return make_strictly_increasing(edges)

def count_bins(
    data: pd.DataFrame,
    bin_edges: Optional[npt.ArrayLike],
    part: Union[int, str],
    congruency: Hashable,
    congruency_condition: Union[int, str],
    accuracy: Hashable,
    accuracy_condition: Union[int, str],
    *,
    id_name: Hashable = "id",
    rt: Hashable = "rt",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bin response times and count observations per bin for a single participant × condition cell.

    This helper constructs a complete RT-bin count table for a specific cell defined by
    `(part, congruency_condition, accuracy_condition)`. It returns a DataFrame with exactly
    `n_bins` rows (rt_bin = 0..n_bins-1) even when some bins contain zero observations.

    The function is designed for quantile-binning workflows (e.g., multinomial likelihood /
    G² deviance computations) where both observed and simulated data must be represented on
    an identical set of RT bin edges.

    Workflow:
    1) Create a full "skeleton" DataFrame (`empty_df`) containing all bin indices for the
       requested cell.
    2) Filter non-finite RT values (NaN, ±inf).
    3) Use `pandas.cut` to assign each RT to an integer bin index based on `bin_edges`.
    4) Count observations in each bin via `groupby(...).count()`.
    5) Left-merge counts into the skeleton and fill missing bins with zeros.

    Parameters
    ----------
    data : pandas.DataFrame
        Trial-level data for (at least) one participant and one congruency × accuracy cell.
        Must contain columns referenced by `id_name`, `congruency`, `accuracy`, and `rt`.
    bin_edges : array-like or None
        Bin edges to use for RT discretization. Typically produced by `get_bin_edges(...)`.
        Must be strictly increasing and compatible with `pandas.cut`.
        If None, or if `data` has zero rows, the function returns zero counts for all bins.
    part : int or str
        Participant identifier to populate in the returned count table. This value is written
        into the `id_name` column for all returned rows.
    congruency : Hashable
        Column name in `data` which stores congruency labels (e.g., "congruent"/"incongruent").
        Also used as the column name in the returned DataFrame.
    congruency_condition : int or str
        Condition value to populate in the returned congruency column for all rows (e.g.,
        "congruent").
    accuracy : Hashable
        Column name in `data` which stores accuracy coding (e.g., 0/1).
        Also used as the column name in the returned DataFrame.
    accuracy_condition : int or str
        Accuracy value to populate in the returned accuracy column for all rows (e.g., 1 for
        correct, 0 for error).
    id_name : Hashable, optional
        Column name for participant IDs in `data` and in the returned DataFrame.
        Default is "id".
    rt : Hashable, optional
        Column name in `data` containing response times to be binned. Default is "rt".
    n_bins : int, optional
        Number of RT bins expected (i.e., the number of intervals implied by `bin_edges`).
        The returned DataFrame will contain exactly `n_bins` rows with `rt_bin = 0..n_bins-1`.
        Default is 10.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns `[id_name, congruency, accuracy, "rt_bin", "obs_count"]`
        and exactly `n_bins` rows. `obs_count` is an integer count (may be returned as float
        after merge/fill operations; cast if you need strict integer dtype).

    Notes
    -----
    - RT values outside the interior edges are still assigned a bin because `bin_edges`
      is typically bounded by [-inf, +inf].
    - Trials whose RT cannot be binned (e.g., due to malformed edges) are dropped via
      `dropna(subset=["rt_bin"])`.
    - This function does not validate that `bin_edges` implies `n_bins`; ensure consistency
      upstream (e.g., `n_bins = len(bin_edges) - 1`).

    Examples
    --------
    >>> edges = get_bin_edges(obs_cell["rt"])  # [-inf, q1, ..., q9, inf]
    >>> counts = count_bins(
    ...     data=obs_cell,
    ...     bin_edges=edges,
    ...     part=12,
    ...     congruency="congruency",
    ...     congruency_condition="incongruent",
    ...     accuracy="accuracy",
    ...     accuracy_condition=1,
    ...     n_bins=10,
    ... )
    >>> counts.head()
       id    congruency  accuracy  rt_bin  obs_count
    0  12  incongruent         1       0          3
    1  12  incongruent         1       1          4
    """
    empty_df = pd.DataFrame(
        {
            id_name: part,
            congruency: congruency_condition,
            accuracy: accuracy_condition,
            "rt_bin": range(n_bins),
        }
    )

    if data.shape[0] == 0 or bin_edges is None:
        empty_df["obs_count"] = 0
        return empty_df

    data = data.copy()
    data = data[np.isfinite(data[rt])]

    data["rt_bin"] = pd.cut(data[rt], bins=bin_edges, labels=False, include_lowest=True)
    data = data.dropna(subset=["rt_bin"])
    data["rt_bin"] = data["rt_bin"].astype(int)

    count_data = (
        data.groupby([id_name, congruency, accuracy, "rt_bin"])[rt]
        .count()
        .reset_index(name="obs_count")
    )

    return (
        empty_df.merge(
            count_data, on=[id_name, congruency, accuracy, "rt_bin"], how="left"
        )
        .fillna(0)
    )

def goodness_of_fit(
    observed: npt.ArrayLike,
    expected: npt.ArrayLike,
    statistic: Literal["g2", "chi2", "both"] = "both",
) -> float | tuple[float, float]:
    """
    Compute multinomial goodness-of-fit statistics (G² and/or Pearson χ²).

    This function evaluates the discrepancy between observed counts (O_j)
    and model-implied expected counts (E_j) using:

        - Likelihood-ratio deviance (G²):
            G² = 2 Σ O_j log(O_j / E_j)

        - Pearson chi-square (χ²):
            χ² = Σ (O_j − E_j)² / E_j

    Both statistics are asymptotically chi-square distributed under
    regularity conditions, with degrees of freedom equal to the number
    of independent cells minus the number of fitted parameters.

    Parameters
    ----------
    observed : array-like
        One-dimensional array of non-negative observed counts (O_j).
    expected : array-like
        One-dimensional array of strictly positive expected counts (E_j).
        Must have the same shape as `observed`.
    statistic : {"g2", "chi2", "both"}, optional
        Which statistic to return:
            - "g2"   → return G² only 
            - "chi2" → return Pearson χ² only
            - "both" → return (G², χ²) (default)

    Returns
    -------
    float or tuple of float
        The requested goodness-of-fit statistic(s).
        Returns np.inf if any E_j <= 0 where O_j > 0 (G² undefined)
        or if any E_j <= 0 (χ² undefined).

    Raises
    ------
    ValueError
        If `observed` and `expected` differ in shape.

    Notes
    -----
    - Cells with O_j = 0 contribute 0 to G².
    - Pearson χ² includes all cells (including O_j = 0).
    - No continuity correction is applied.
    - Assumes multinomial count structure.

    Examples
    --------
    >>> goodness_of_fit([10, 15, 5], [12, 12, 6], "g2")
    1.527...

    >>> goodness_of_fit([10, 15, 5], [12, 12, 6], "chi2")
    1.583...

    >>> goodness_of_fit([10, 15, 5], [12, 12, 6], "both")
    (1.527..., 1.583...)
    """
    O = np.asarray(observed, dtype=float)
    E = np.asarray(expected, dtype=float)

    if O.shape != E.shape:
        raise ValueError("`observed` and `expected` must have the same shape.")

    if np.any(E <= 0):
        return float(np.inf) if statistic != "both" else (float(np.inf), float(np.inf))

    # --- G² ---
    mask = O > 0
    G2 = 2.0 * np.sum(O[mask] * np.log(O[mask] / E[mask]))

    # --- Pearson χ² ---
    chi2 = np.sum((O - E) ** 2 / E)

    if statistic == "g2":
        return float(G2)
    elif statistic == "chi2":
        return float(chi2)
    elif statistic == "both":
        return float(G2), float(chi2)
    else:
        raise ValueError("`statistic` must be one of {'g2', 'chi2', 'both'}.")

def compute_gof(
    data_obs: pd.DataFrame,
    data_model: pd.DataFrame,
    *,
    rt: Hashable = "rt",
    congruency: Hashable = "congruency",
    accuracy: Hashable = "accuracy",
    min_n_err: int = 5,
    id_name: Hashable = "id",
    n_bins: int = 10,
    E_min: int = 1
) -> npt.NDArray[np.float64]:
    """
    Compute cell-wise multinomial G² deviances between observed and model data.

    This function evaluates model fit by comparing observed and simulated
    response-time (RT) distributions within each participant × congruency ×
    accuracy cell. RTs are discretized into quantile-based bins derived from
    the observed data, and a multinomial likelihood-ratio deviance (G²) is
    computed for each cell.

    For each participant and each condition combination:

        1. Empirical RT quantiles define bin edges.
        2. Observed and simulated RTs are binned using identical edges.
        3. Observed bin counts (O) are compared to model-implied expected
           counts (E) derived from simulated counts using Dirichlet smoothing.
        4. A G² deviance statistic is computed.

    The function returns a vector of G² values across all evaluable cells.

    Parameters
    ----------
    data_obs : pandas.DataFrame
        Observed trial-level dataset. Must contain at least:
        - participant identifier column (`id_name`)
        - congruency column (`congruency`)
        - accuracy column (`accuracy`)
        - RT column (`rt`)
    data_model : pandas.DataFrame
        Model-generated (simulated) trial-level dataset with the same column
        structure as `data_obs`.
    rt : Hashable, optional
        Column name for response times. Default is "rt".
    congruency : Hashable, optional
        Column name for congruency condition labels. Default is "congruency".
    accuracy : Hashable, optional
        Column name for accuracy coding (e.g., 0 = error, 1 = correct).
        Default is "accuracy".
    min_n_err : int, optional
        Minimum number of observed error trials required to compute a
        meaningful error RT distribution. Cells with fewer error trials
        are skipped. Default is 5.
    id_name : Hashable, optional
        Column name identifying participants. Default is "id".
    n_bins : int, optional
        Number of RT bins (typically quantile-based). Must be consistent
        with the output of `get_bin_edges`. Default is 10.
    E_min : int
        Minimum expected values in a bin. Is used to filter overestimated Chi² values. Default is 1.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of G² deviance values (float64), one per
        evaluable participant × congruency × accuracy cell.

    Notes
    -----
    - Quantile bin edges are computed from observed RTs only.
    - Model bin probabilities are estimated from simulated counts with
      Dirichlet smoothing (α = 0.5).
    - If the model produces zero simulated trials in a cell, the cell
      is skipped rather than forcing infinite deviance.
    - Cells with insufficient observed error trials (if accuracy == 0)
      are skipped.
    - The resulting G² values correspond to multinomial deviances on
      binned RT distributions, not to the continuous-time diffusion
      model likelihood.

    Statistical Interpretation
    --------------------------
    For each cell, the deviance is:

        G² = 2 * Σ O_j log(O_j / E_j),

    where O_j are observed bin counts and E_j are expected counts implied
    by the model. Under regularity conditions and large samples, G² is
    asymptotically chi-square distributed.

    Examples
    --------
    >>> g2_values = compute_g2(
    ...     data_obs=empirical_df,
    ...     data_model=simulated_df,
    ...     n_bins=10
    ... )
    >>> g2_values.mean()
    12.47
    """
    check_vars(data=data_obs, id_name=id_name, rt=rt,
               congruency=congruency, accuracy=accuracy)
    check_vars(data=data_model, id_name=id_name, rt=rt,
               congruency=congruency, accuracy=accuracy)

    data_obs = check_congruency(
        data=data_obs,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    data_model = check_congruency(
        data=data_model,
        rt=rt,
        congruency=congruency,
        output_coding_con="congruent",
        output_coding_inc="incongruent",
    )

    parts = data_obs[id_name].unique()
    rows = []

    for idx in tqdm(range(0, len(parts)), desc=f"Compute Goodness of Fit"):
        for con in ["congruent", "incongruent"]:
            for acc in [0, 1]:

                part = parts[idx]

                obs_cell = data_obs[
                    (data_obs[id_name] == part)
                    & (data_obs[congruency] == con)
                    & (data_obs[accuracy] == acc)
                ].copy()

                mod_cell = data_model[
                    (data_model[id_name] == part)
                    & (data_model[congruency] == con)
                    & (data_model[accuracy] == acc)
                ].copy()

                if obs_cell.shape[0] == 0:
                    continue

                bin_edges = get_bin_edges(obs_cell[rt])

                count_obs = count_bins(
                    obs_cell,
                    bin_edges,
                    part,
                    id_name=id_name,
                    congruency=congruency,
                    accuracy=accuracy,
                    congruency_condition=con,
                    accuracy_condition=acc,
                    n_bins=n_bins,
                )

                count_mod = count_bins(
                    mod_cell,
                    bin_edges,
                    part,
                    id_name=id_name,
                    congruency=congruency,
                    accuracy=accuracy,
                    congruency_condition=con,
                    accuracy_condition=acc,
                    n_bins=n_bins,
                )

                merged = (
                    pd.merge(
                        count_obs,
                        count_mod,
                        on=[id_name, congruency, accuracy, "rt_bin"],
                        suffixes=["_obs", "_model"],
                        how="outer",
                    )
                    .fillna(0)
                )

                O = merged["obs_count_obs"].to_numpy(dtype=float)
                C = merged["obs_count_model"].to_numpy(dtype=float)

                N_obs = O.sum()
                N_sim = C.sum()

                if acc == 0 and N_obs < min_n_err:
                    continue

                if N_sim == 0:
                    continue

                alpha = 0.5
                J = len(C)

                pi = (C + alpha) / (N_sim + J * alpha)
                E = N_obs * pi

                G2, chi2 = goodness_of_fit(O, E, statistic="both")

                if np.min(E) < E_min:
                    chi2 = np.nan

                rows.append({
                    str(id_name): part,
                    str(congruency): con,
                    str(accuracy): acc,
                    "n_obs": float(N_obs),
                    "n_sim": float(N_sim),
                    "g2": float(G2),
                    "chi2": float(chi2),
                    "min_E": float(np.min(E)),
                })

    return pd.DataFrame(rows)