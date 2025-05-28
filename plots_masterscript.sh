#!/bin/bash

# List of Python files
files=("posterior_predictive_check_delta_functions.py" "posterior_predictive_check.py" "experimental_effects.py")

# "metrics_num_obs.py"
# "diagnostics.py"

# if two networks are compared, choose the fixed sdr as network_name and the estimated as ...2
network_name="dmc_optimized_winsim_priors_sdr_fixed_200_805382"
num_obs=300
host="local"
num_obs=300
network_name2="dmc_optimized_winsim_priors_sdr_estimated_200_805375"
repetitions=1000
data_sets=100
num_resims=100

eval "$(conda shell.bash hook)"
conda activate bf_new

export PYTHONPATH="/home/administrator/Documents/bf_dmc:$PYTHONPATH"

# Loop over each file and execute
for file in "${files[@]}"
do
    echo "Running $file..."
    python "plot_scripts/$file" "$network_name" "$host" "$num_obs" "$network_name2" "$repetitions" "$data_sets" "$num_resims"# Add any extra args you want here
done

echo "All scripts executed."
