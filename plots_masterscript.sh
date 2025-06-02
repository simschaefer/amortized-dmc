#!/bin/bash

# List of Python files
#files=("posterior_predictive_check_delta_functions.py" "posterior_predictive_check.py" "experimental_effects.py")

#files=("diagnostics.py")

files=("prior_predictive_check.py")

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <network_name>"
    exit 1
fi

# Get the network name from the argument
network_name="$1"
network_name2="$2"

# "metrics_num_obs.py"
# "diagnostics.py"

# if two networks are compared, choose the fixed sdr as network_name and the estimated as ...2
num_obs=200
host="local"
#network_name2="dmc_optimized_winsim_priors_sdr_estimated_200_810183"
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
    python "plot_scripts/$file" "$network_name" "$host" "$num_obs" "$network_name2" "$repetitions" "$data_sets" "$num_resims"       
done

echo "All scripts executed."
