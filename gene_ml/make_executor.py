from GENE_ML.gene_ml.samplers.uniform import Uniform
from GENE_ML.gene_ml.executors.ScanExecutor import ScanExecutor
from GENE_ML.gene_ml.runners.GENErunner import GENErunner
from GENE_ML.gene_ml.parsers.GENEparser import GENE_scan_parser

import os

def make_executor(config, sampler, remote_save_name, guess_sample_wallseconds, num_workers):
    remote_save_dir = os.path.join(config.remote_save_base_dir,remote_save_name)
    parser = GENE_scan_parser(config.save_dir, config.base_params_path, remote_save_dir)
    runner = GENErunner(parser, config.host, config.sbatch_base_path, guess_sample_wallseconds, config.remote_run_dir)

    ##Executor
    #The executor will divide the samples into batches; one for each worker. Each batch will be ran in paralell in seperate sbatch jobs. 
    # The executor should alter a base batch script to account for that less samples will be ran. 
    # num_workers = 2
    executor = ScanExecutor(num_workers, sampler, runner)
    return executor, remote_save_dir


if __name__ == "__main__":

    parameters = ['box-kymin', '_grp_species_1-omt', '_grp_species_0-omt', 'species-omn', 'geometry-q0', 'geometry-shat', 'general-beta']
    bounds = [(0.05,1), (10,70), (10,70), (5,60), (3,6), (0.001,4), (0.1E-2,0.4E-2)]
    num_samples = 40
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    
    import sys
    sys.path.append(os.path.join('home','djdaniel','GENE_UQ','config'))
    from config import config
    make_executor(config ,sampler, remote_save_name='test', guess_sample_wallseconds=200, num_workers=2)




