import numpy as np
import sys
sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/')    
import os
import time

class ScanExecutor():
    def __init__(self, num_workers, sampler, runner, ex_id, **kwargs):
        self.num_workers = num_workers  # kwargs.get('num_workers')
        self.sampler = sampler
        self.runner = runner
        self.remote_save_dir = self.runner.parser.remote_save_dir
        self.ex_id = ex_id
        self.sbatch_ids = [] #the sbatch id is the slurm sbatch id
        self.batches = {k:np.array_split(v,self.num_workers) for k,v in self.sampler.samples.items()}
        self.batches = [{k:v[i] for k,v in self.batches.items()} for i in range(self.num_workers)]
        self.run_ids =  [f'ex-{self.ex_id}_batch-{b_id}' for b_id in np.arange(len(self.batches))] #the runid will be the name of the folder in the remote_run_dir
        
    
    def start_runs(self):
        print(100 * "=")
        print("EXECUTING BATCHES")
        #self.batches is a list of dictionaries where each one is a subset of the entire samples dictionary.
        for batch, rid in zip(self.batches, self.run_ids):
            self.runner.parser.remote_save_dir = os.path.join(self.remote_save_dir, f'{rid}')
            self.sbatch_ids.append(self.runner.code_run(batch, run_id=rid))

    def check_finished(self):
        #To check that the executors sbatch id's are no longer in the squeue
        print('BATCH IDS', self.sbatch_ids)
        finished = self.runner.check_finished(self.sbatch_ids)
        print('EXECUTOR CHECK FINISHED', finished)
        return finished
    
    def kill(self):
        self.runner.kill_runs(self.sbatch_ids)
    
    def check_complete(self):
        #To check that all the runs have been complete and a continue scan is not needed
        incomplete = self.runner.check_complete(self.run_ids)
        print('EXECUTOR CHECK COMPLETE', incomplete)
        return incomplete
    
    def continue_run(self):
        incomplete = self.check_complete()
        if len(incomplete)==0:
            print('ALL RUNS HAVE BEEN COMPLETED OR FAILED')
            return []
        sbatch_ids = self.runner.continue_run(incomplete)
        self.sbatch_ids = self.sbatch_ids + sbatch_ids
        return sbatch_ids

    def continue_n_times(self, n, check_every=30):
        for i in range(n):
            print('EXECUTING CONTINUE NUMBER:', i)
            start = time.time()
            while not self.check_finished():
                now = time.time()
                print('Waiting untill the runs are finished. TIMER: ', self.runner.sec_to_time_format(now-start), ' dd-hh:mm:ss')
                time.sleep(check_every)
            print('RUNS ARE FINISHED')

            continue_result = self.continue_run()
            if len(continue_result) == 0:
                break #all runs have finished or failed, there is nothing to continue
        print('END OF CONTINUE N TIMES')

    def delete(self):
        print('EXECUTOR DELETE')
        self.runner.delete(self.run_ids)
        self.runner.delete_remote_dir(self.remote_save_dir)
        print('EXECUTOR DELETING run_files')
        for sbatch_id in self.sbatch_ids:
            os.system(f'rm run_files/auto_gene.{sbatch_id}.out')
    
if __name__ == '__main__':
    import os
    import sys
    sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/gene_ml')
    from samplers.uniform import Uniform
    from runners.GENErunner import GENErunner
    from parsers.GENEparser import GENE_scan_parser
    parameters = ['box-kymin', '_grp_species_1-omt', 'species-omn']
    bounds = [(0.05,1), (10,70), (5,60)]
    num_samples = 10
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    base_params_path = os.path.join(os.getcwd(),'parameters_base_dp')
    parser = GENE_scan_parser(base_params_path)
    runner = GENErunner(parser, remote_run_dir='/project/project_462000451/gene_auto/', host='lumi', sbatch_base_path='sbatch_base_dp', single_run_timelim=81)
    sbatch_base_path='/home/djdaniel/DEEPlasma/sbatch_base_dp'
    executor = ScanExecutor(num_workers=5 ,sampler=sampler, runner=runner)
    executor.start_runs()
