conda create -n gene_ml -c conda-forge -c pytorch -c anaconda optuna ipykernel matplotlib pandas numpy scikit-learn tqdm nb_conda_kernels f90nml
conda activate gene_ml
pip install GPy==1.13.1
conda env export --name gene_ml > gene_ml.yml
