import numpy as np
import sys
sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/')    
import os
import time
from ..tools import sec_to_time_format

class ScanExecutor():
    def __init__(self, config, num_workers, sampler, runner, ex_id, **kwargs):
        self.config = config
        self.num_workers = num_workers  # kwargs.get('num_workers')
        self.sampler = sampler
        self.runner = runner
        self.ex_id = ex_id
        self.sbatch_ids = [] #the sbatch id is the slurm sbatch id
        self.batches = {k:np.array_split(v,self.num_workers) for k,v in self.sampler.samples.items()}
        self.batches = [{k:v[i] for k,v in self.batches.items()} for i in range(self.num_workers)]
        self.run_ids =  [f'ex-{self.ex_id}_batch-{b_id}' for b_id in np.arange(len(self.batches))] #the runid will be the name of the folder in the remote_run_dir
        
    def pre_run_check(self):
        print('\nEXECUTOR, PRINT_CHECK_PARAMETERS\n', 100*'-')
        self.runner.pre_run_check(self.batches[0],self.run_ids[0])

    def start_runs(self):
        print(100 * "=")
        print("EXECUTING BATCHES")
        #self.batches is a list of dictionaries where each one is a subset of the entire samples dictionary.
        for batch, rid in zip(self.batches, self.run_ids):
            self.sbatch_ids.append(self.runner.code_run(batch, run_id=rid))
        

    def check_finished(self):
        #To check that the executors sbatch id's are no longer in the squeue
        finished = self.runner.check_finished(self.sbatch_ids)
        print('EXECUTOR, CHECK FINISHED', finished)
        return finished
    
    def wait_till_finished(self, check_interval=5):
        self.runner.wait_till_finished(self.sbatch_ids, check_interval)

    def kill(self):
        self.runner.kill_runs(self.sbatch_ids)
    
    def check_complete(self):
        #To check that all the runs have been complete and a continue scan is not needed
        complete = self.runner.check_complete(self.run_ids)
        print('EXECUTOR, LIST OF COMPLETE SBATCH IDs', complete)
        if all(complete):
            print('ALL SBATCH IDs ARE COMPLETE FOR THIS EXECUTOR')
        else:
            print('EXECUTOR, LIST OF INCOMPLETE RUN IDs', np.array(self.run_ids)[~complete])
        return complete
    
    def continue_incomplete(self):
        complete = self.check_complete()
        if all(complete):
            print('ALL RUNS HAVE BEEN COMPLETED')
            return []
        sbatch_ids = []
        for run_id in self.run_ids[not complete]:
            sbatch_ids.append(self.runner.continue_run(run_id))
        self.sbatch_ids = self.sbatch_ids + sbatch_ids
        return sbatch_ids

    def delete(self):
        print('EXECUTOR DELETE')
        self.runner.delete(self.run_ids)
        self.runner.delete_remote_dir(self.remote_save_dir)
        print('EXECUTOR DELETING run_files')
        for sbatch_id in self.sbatch_ids:
            os.system(f'rm run_files/auto_gene.{sbatch_id}.out')

    
    def local_backup(self):
        for run_id in self.run_ids:
            self.config.scp.get(local_path = self.config.local_backup_dir, remote_path = os.path.join(self.config.remote_run_dir,'auto_prob_'+run_id), recursive=True)
        self.config.scp.get(local_path = self.config.local_backup_dir, remote_path=self.remote_save_dir, recursive=True)

    def continue_non_converged_runs(self, new_simtimelims):
        # n is the number of times to continue
        # This is only to be ran when there is a set of gene problem directories already created by the start_runs function.
        self.wait_till_finished()
        if not all(self.check_complete()):
            raise TimeoutError('Daniel Says: gene_status shows incomplete individual gene runs. This is likely because the wallclock limit was hit. Check how this is calculated')

        for i, simtimelim in enumerate(new_simtimelims):
            print('CONTINUING NON CONVERGED RUNS',i)
            self.runner.continue_non_converged_runs(self.run_ids, new_simtimelim=simtimelim)
        print(f'NON CONVERGED RUNS HAVE BEEN CONTINUED {i} TIMES')
        print(f'NEW SIMTIMELIM FOR RECENT CONTINUE IS {simtimelim}')


    # def increment_sim_time_lim(self, simtimelims):
    #     # This is only to be ran when there is a set of gene problem directories already created by the start_runs function.
    #     # Check to see there is nothing running that was started by this executor
    #     self.wait_till_finished()
    #     if not all(self.check_complete()):
    #         raise TimeoutError('Daniel Says: gene_status shows incomplete individual gene runs. This is likely because the wallclock limit was hit. Check how this is calculated')
    #     for simtimelim in simtimelims:
    #         print('STARTING NEXT SIMTIMELIM:',simtimelim, 'out of',simtimelims)
    #         self.sbatch_ids += self.runner.continue_with_increased_simtimelim(self.run_ids,simtimelim)
    #         self.wait_till_finished()
        
    #     print('FINISHED INCRIMENTING SIM TIME LIM')

    def continue_increment(self, group_var, values):
        self.wait_till_finished()
        for value in values:
            self.sbatch_ids += self.runner.continue_with_new_param(self.run_ids, group_var, value)
            self.wait_till_finished()
            latest_scanfiles = self.runner.get_latest_scanfiles_path(self.run_ids)
            for latest_sf in latest_scanfiles:
                self.runner.parser.rename_important_scanfiles(latest_sf, prefix='-'.join(group_var)+'_value')
    

    
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
    runner = GENErunner(parser, remote_run_dir='/project/project_462000451/gene_auto/', host='lumi', base_sbatch_path='sbatch_base_dp', single_run_timelim=81)
    base_sbatch_path='/home/djdaniel/DEEPlasma/sbatch_base_dp'
    executor = ScanExecutor(num_workers=5 ,sampler=sampler, runner=runner)
    executor.start_runs()
