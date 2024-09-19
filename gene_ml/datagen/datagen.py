from GENE_ML.gene_ml.samplers.uniform import Uniform
from GENE_ML.gene_ml.executors.ScanExecutor import ScanExecutor
from GENE_ML.gene_ml.runners.GENErunner import GENErunner
from GENE_ML.gene_ml.parsers.GENEparser import GENE_scan_parser
from time import sleep
import os

class DataGen():
    def __init__(self, config, sampler, remote_save_name, num_workers, ex_id, single_run_timelim=None, previous_set=None, time_model=None):
        self.remote_save_name = remote_save_name
        self.remote_save_dir = os.path.join(config.remote_save_base_dir,remote_save_name)
        self.parser = GENE_scan_parser(config.save_dir, config.base_params_path, self.remote_save_dir)
        self.single_run_timelim = single_run_timelim
        self.ex_id = ex_id
        #switching between using a time_model or guessing the walltime for a sample
        if type(previous_set) != type(None):
            previous_set.train_time_model(time_model)
            self.runner = GENErunner(self.parser, config.host, config.sbatch_base_path, config.remote_run_dir, time_model=previous_set.time_model, local_run_files_dir=config.local_run_files_dir)
        else:
            self.runner = GENErunner(self.parser, config.host, config.sbatch_base_path, config.remote_run_dir, single_run_timelim=single_run_timelim, local_run_files_dir=config.local_run_files_dir)
        
        self.sampler = sampler

        ##Executor
        #The executor will divide the samples into batches; one for each worker. Each batch will be ran in paralell in seperate sbatch jobs. 
        # The executor should alter a base batch script to account for that less samples will be ran. 
        # num_workers = 2
        self.executor = ScanExecutor(num_workers, sampler, self.runner, ex_id)
    


    def run_and_recieve(self, test_percentage=0):
        # Not functionining, need to change scan data so that it can look at the correct directory
        self.executor.start_runs()
        i = 1
        while not self.executor.check_finished():
            sleep(1)#self.single_run_timelim)
            print(f'check {i}')
            i+=1
        from GENE_ML.gene_ml.dataset.ScanData import ScanData
        from config import config
        local_save_name = self.remote_save_name
        self.data_set = ScanData(local_save_name, self.parser, config.host, remote_path=self.remote_save_dir, test_percentage=test_percentage)
        return self.data_set

if __name__ == "__main__":

    parameters = ['box-kymin', '_grp_species_1-omt', '_grp_species_0-omt', 'species-omn', 'geometry-q0', 'geometry-shat', 'general-beta']
    bounds = [(0.05,1), (10,70), (10,70), (5,60), (3,6), (0.001,4), (0.1E-2,0.4E-2)]
    num_samples = 40
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    
    import sys
    sys.path.append(os.path.join('home','djdaniel','GENE_UQ','config'))
    from config import config
    make_executor(config ,sampler, remote_save_name='test', single_run_timelim=200, num_workers=2)




