#!/usr/bin/env bash
set -euo pipefail

notebooks=(
  "$HOME/Documents/gits/igosip/analyses/maxabsversions/SIMULATION-TCGA_high_to_low_risk-newWheel.ipynb"
  "$HOME/Documents/gits/igosip/analyses/maxabsversions/SIMULATION-TCGA_high_to_low_risk.ipynb"
  "$HOME/Documents/gits/igosip/analyses/maxabsversions/SIMULATION-TCGA_high_to_low_risk-KIRC.ipynb"
  "$HOME/Documents/gits/igosip/analyses/maxabsversions/gosip_vae_pan_tcga_all_cancers_FINALabs.ipynb"
)

for nb in "${notebooks[@]}"; do
  echo "Executing $nb ..."
  jupyter nbconvert \
    --to notebook \
    --execute "$nb" \
    --output "${nb%.ipynb}.executed.ipynb" 
  echo "Completed $nb"
done
