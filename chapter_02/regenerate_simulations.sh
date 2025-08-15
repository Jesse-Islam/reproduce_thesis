#!/bin/bash

# Define the list of notebooks to run

main_path="/home/jislam/Documents/gits/reproduce_thesis/chapter_02/analyses"

# Define the list of notebook filenames (without the path)
notebooks=(
    "vae_sim_realistic_500_control_alpha_to_beta_custom.ipynb"
    "vae_sim_realistic_500_control_alpha_to_beta.ipynb"
    "vae_sim_realistic_500_beta_T2D_to_control.ipynb"
)

# Loop through each notebook and execute it
for notebook in "${notebooks[@]}"; do
    full_path="$main_path/$notebook"  # Prepend main_path to the filename
    echo "Running $full_path..."
    
    # Run the notebook and stop the script if an error occurs
    jupyter nbconvert --to notebook --execute "$full_path" --output "$full_path"

    if [ $? -ne 0 ]; then
        echo "Error executing $full_path. Stopping script."
        exit 1
    fi

    echo "Finished $full_path."
done

echo "All notebooks executed successfully!"
