#navigate to the directory of this folder in the terminal and run:
#docker build -t gene_ml_image .
#this will make the docker image gene_ml_image which can be used to run the container with the docker_make.sh file.

FROM continuumio/miniconda3
# I had issues with file not found when trying to get the run to work. This copy was the only fix I could get to make it work.
#COPY <host_src> <container_dest>
COPY gene_ml_conda.sh /home/djdaniel/DEEPlasma/GENE_ML/conda_docker/gene_ml_conda.sh
COPY docker_ssh_config /root/.ssh/config 
VOLUME /home/djdaniel/
WORKDIR /home/djdaniel/
# RUN conda create -n gene_ml -c conda-forge -c pytorch -c anaconda optuna ipykernel matplotlib pandas numpy scikit-learn tqdm nb_conda_kernels f90nml xgboost -y
RUN /home/djdaniel/DEEPlasma/GENE_ML/conda_docker/gene_ml_conda.sh
# RUN pip install GPy==1.13.1
RUN apt-get update && apt-get install vim -y
