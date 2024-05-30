#!/bin/sh

#Run this to create a conda enviroment for gene_ml projects and export the enviorment to a yml file
conda create -n gene_ml -c conda-forge -c pytorch -c anaconda optuna chaospy ipykernel matplotlib pytorch pandas numpy scikit-learn tqdm nb_conda_kernels f90nml xgboost -y
conda activate gene_ml
pip install GPy==1.13.1
