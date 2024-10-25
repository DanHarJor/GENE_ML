from GENE_ML.gene_ml.samplers.uniform import Uniform
from GENE_ML.gene_ml.executors.ScanExecutor import ScanExecutor
from GENE_ML.gene_ml.runners.GENErunner import GENErunner
from GENE_ML.gene_ml.parsers.GENEparser import GENE_scan_parser
from time import sleep
import os

class DataGen():
    def __init__(self, config, sampler, remote_save_name, num_workers, ex_id, single_run_timelim=None, single_run_simtimelim=None, previous_set=None, time_model=None):
        self.remote_save_name = remote_save_name
        self.remote_save_dir = os.path.join(config.remote_save_base_dir,remote_save_name)
        self.parser = GENE_scan_parser(config)
        self.single_run_timelim = single_run_timelim
        self.ex_id = ex_id
        #switching between using a time_model or guessing the walltime for a sample
        if type(previous_set) != type(None):
            previous_set.train_time_model(time_model)
            self.runner = GENErunner(self.parser, config, self.remote_save_dir, single_run_timelim=single_run_timelim, single_run_simtimelim=single_run_simtimelim, time_model=previous_set.time_model)
        
        else:
            self.runner = GENErunner(self.parser, config, self.remote_save_dir, single_run_timelim=single_run_timelim, single_run_simtimelim=single_run_simtimelim)
        
        self.sampler = sampler

        ##Executor
        #The executor will divide the samples into batches; one for each worker. Each batch will be ran in paralell in seperate sbatch jobs. 
        # The executor should alter a base batch script to account for that less samples will be ran. 
        # num_workers = 2
        self.executor = ScanExecutor(config, num_workers, sampler, self.runner, ex_id, self.remote_save_dir)
    


if __name__ == "__main__":

    parameters = ['box-kymin', '_grp_species_1-omt', '_grp_species_0-omt', 'species-omn', 'geometry-q0', 'geometry-shat', 'general-beta']
    bounds = [(0.05,1), (10,70), (10,70), (5,60), (3,6), (0.001,4), (0.1E-2,0.4E-2)]
    num_samples = 40
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    



