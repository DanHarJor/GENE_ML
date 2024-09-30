import os
import sys
import subprocess
import numpy as np
from collections.abc import Iterator
import re
from ..tools import sec_to_time_format

class GENErunner():
    def __init__(self, parser, config, time_model=None, single_run_timelim=None, single_run_simtimelim=None):
        self.parser=parser
        self.host = config.host
        self.single_run_timelim = single_run_timelim
        self.single_run_simtimelim = single_run_simtimelim
        self.remote_run_dir = config.remote_run_dir
        self.ssh_path = f"{self.host}:{self.remote_run_dir}"
        self.time_model = time_model
        self.local_run_files_dir = config.local_run_files_dir
        self.max_wallseconds = 0
        self.config = config

    def continue_run(self, run_id, purpose='not_all_finished'):
        # if 
        print('CONTINUING RUNS -', run_id)
        if type (run_id) is list:
            sbatch_ids = []
            for rid in run_id:
                command = f"ssh {self.host} 'cd {os.path.join(self.remote_run_dir, 'auto_prob_'+rid)}; sbatch continue.cmd; exit'"
                sbatch_id = subprocess.check_output(command, shell=True, text=True)
                sbatch_id = sbatch_id.strip().split(' ')[-1]
                sbatch_ids.append(sbatch_id)
            return sbatch_ids
        else:
            raise TypeError("run_id must be a list, for only one run_id please place into a list.")

    def get_wallseconds(self, samples):
        #using time model or guess sample walltime to get the walltime
        if type(self.time_model)!=type(None):
            times, errors = self.time_model.predict(samples)
            wallseconds = np.sum(times) * 1.3
        else:
            n_samples = len(list(samples.values())[0])
            print('\n\nSINGLE RUN TIMELIM',self.single_run_timelim, 'N SAMPLES', n_samples)
            wallseconds = self.single_run_timelim * n_samples * 1.1 #add 10% more to ensure it works
        return wallseconds

    def pre_run_check(self, samples, run_id):
        print('PRE RUN CHECK')
        n_samples = len(list(samples.values())[0])
        wallseconds = self.get_wallseconds(samples)
        if wallseconds > self.max_wallseconds: self.max_wallseconds = wallseconds
        print(f"THE ESTIMATED WALLTIME FOR RUN {run_id} is {sec_to_time_format(wallseconds)}, dd-hh-mm-ss TO RUN {n_samples} SAMPLES")
        
        remote_problem_dir = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}') 
        remote_param_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'parameters')
        remote_sbatch_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'submit.cmd')
        remote_continue_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'continue.cmd')
    
        try:
            print(f"CHECKING IF PROBLEM DIRECTORY EXISTS?")
            directory_details = self.config.paramiko_sftp_client.stat(remote_problem_dir)
        except FileNotFoundError:
            print('REMOTE PROBLEM DIRECTORY DOES NOT EXIST, CREATING IT NOW:', remote_problem_dir)
            command = f'cd {self.remote_run_dir} && ./newprob && mv prob01 auto_prob_{run_id}; exit'
            _, stdout, _ = self.config.paramiko_ssh_client.exec_command(command)
            print('RESULT FROM COMMAND:',stdout.read())

        self.parser.base_to_remote(remote_param_path, remote_sbatch_path)

        print(f"ALTERING THE PARAMETERS FILE IN THE REMOTE PROBLEM DIRECTORY")
        self.parser.alter_parameters_file(remote_param_path, group_var=["general","timelim"], value=self.single_run_timelim) # If using time_model this should be set with the time model in the sampler and put in the scan format on the parameters file.
        
        print('\n\nCODE RUN: SETTING SIMULATION TIME LIMMIT\n\n')
        # simtimelim is the timelimit inside the simulation, so number of seconds of plasma evolution. The simulation should be fater than walltime so I set it to the same time to ensure no limitations here.
        self.parser.set_simtimelim(self.single_run_simtimelim, parameters_path=remote_param_path)

        print('SBATCH')
        print(self.parser.write_sbatch(remote_sbatch_path, remote_continue_path, wallseconds))

        print(f'PARSING SAMPLES TO INPUT FILE at:',remote_param_path)
        print(self.parser.write_input_file(samples, remote_param_path))

    def code_run(self, samples, run_id):
        self.pre_run_check(samples, run_id)

        print('\nCODE RUN')
        # make results directory
        command = f'mkdir -p {self.parser.remote_save_dir}'
        stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(command)
        print('RESULT OF MAKING REMOTE SAVE DIRECTORY,', stdout.read())

        remote_problem_dir = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}') 

        run_command = f'cd {self.remote_run_dir}/auto_prob_{run_id}; sbatch submit.cmd; exit'
        stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(run_command)
        out = (stdout.read(), stderr.read())
        sbatch_id = re.search('(?<![\d])\d{7}(?![\d])', out[0])
        print('OUT:', out[0], 'ERROR?:', out[1])
        print('SUBMITTED SBATCH ID:', sbatch_id)
        return sbatch_id
    
    def kill_runs(self, sbatch_ids):
        print('\n KILLING SBATCH IDS',sbatch_ids)
        os.system(f"ssh {self.host} 'scancel {' '.join(sbatch_ids)}'")
        command = f"ssh {self.host} 'squeue --me'"
        queue = subprocess.check_output(command, shell=True, text=True)
        print('SLURM QUEUE AFTER KILL',queue)
        print('\n')

    # Checks to see if the slurm batch jobs are still in the queue. 
    def check_finished(self, sbatch_ids):
        print('\nCHECKING IF JOBS FINISHED:', sbatch_ids)
        #To check that certain sbatch_id's are no longer in the squeue
        command = f"ssh {self.host} 'squeue --me'"
        queue = subprocess.check_output(command, shell=True, text=True)
        lines = queue.split('\n')
        recieved_ids = [l.strip().split(' ')[0] for l in lines[0:]]
        still_running_ids = set(recieved_ids) & set(sbatch_ids)
        if len(still_running_ids)>0:
            print('RUNNING SBATCH IDs: ', still_running_ids)
            finished = False
        else:
            print('NONE OF THE INPUTED SBATCH IDs ARE RUNNING')
            finished  = True
        print('\n')
        return finished
    
    def retrieve_run_file(self, file_name, run_id):
        print("GETTING RUN FILE: ", file_name)
        command = f"scp '{os.path.join(self.ssh_path, 'auto_prob_'+run_id, file_name)}' {os.path.join(self.local_run_files_dir, file_name)}"
        command = f"scp '{os.path.join(self.ssh_path, 'auto_prob_'+run_id, file_name)}' {os.path.join(self.local_run_files_dir, file_name)}"
        
        os.system(command)

    
    # Checks to see if the GENE run needs to be continued
    def check_complete(self, run_ids):
        incomplete = []
        for run_id in run_ids:
            #Trying to use new paramiko method for this.
            # out_path = os.path.join(self.parser.remote_save_dir, run_id)
            # command = f'ls {out_path}'
            # stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(command)
            # lines = stdout.readlines()
            # print('stdout',lines)
            # scan_numbers = [out[-4:] for out in lines]
            # print('SCAN NUMBERS',scan_numbers)
            # geneerr_log_path = os.path.join(self.parser.remote_save_dir, run_id)

            regex_command = "ls | grep -E '[0-9]{7}.err$'"
            command = f"ssh {self.host} 'cd {os.path.join(self.remote_run_dir,'auto_prob_'+run_id)} &&  {regex_command}; exit'"
            files = subprocess.check_output(command, shell=True, text=True)
            files = files.strip().split('\n')
            ran_sbatch_ids = [f.strip().split('.')[-2] for f in files]
            latest = files[np.argmax(ran_sbatch_ids)]
            self.retrieve_run_file(latest, run_id)
            scan_status = self.parser.read_scan_status(os.path.join(self.local_run_files_dir, latest))
            if scan_status == 'needs continuation':
                incomplete.append(run_id)
        return incomplete
    
    def delete(self, run_ids):
        for run_id in run_ids:
            remote_dir = os.path.join(self.remote_run_dir,'auto_prob_'+run_id)
            temp_dir = f"temp/*{run_id}"
            print('DELETING,', remote_dir, temp_dir)
            os.system(f"ssh {self.host} 'rm -r {remote_dir}; exit'")
            os.system(f"rm {temp_dir}")

    def delete_remote_dir(self, remote_dir):
        print('DELETING,', remote_dir)
        os.system(f"ssh {self.host} 'rm -r {remote_dir}; exit'")

        
    def update_gene_status(self):
        gene_status_path = os.path.join(self.parser.remote_save_dir(),'scanfiles0000','in_pars','gene_status')
        

    def continue_with_increased_omega_prec():
        None

        


    def clean(self):
        '''
            Removes any prob0* directories created by ./newprob and the auto_prob_* directories created for the specific run.
        '''
        print('CLEANING RUN DIR OF RUNER CREATED DIRECTORIES')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && rm -r auto_prob_* prob0*'")

        os.system('rm temp/*')

        

if __name__ == '__main__':
    import os
    sys.path.append(os.path.join('/home/djdaniel/DEEPlasma/GENE_ML/gene_ml'))
    from samplers.uniform import Uniform
    from parsers.GENEparser import GENE_scan_parser
    
    parameters = ['box-kymin', '_grp_species_1-omt', 'species-omn']
    bounds = [(0.05,1), (10,70), (5,60)]
    num_samples = 3
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    base_params_path = os.path.join('/home/djdaniel/DEEPlasma/','parameters_base_dp')
    remote_save_dir='/scratch/project_462000451/gene_out/gene_auto/test'
    parser = GENE_scan_parser(base_params_path, remote_save_dir)
    runner = GENErunner(parser, remote_run_dir='/project/project_462000451/gene_auto/', host='lumi', base_sbatch_path = '/home/djdaniel/DEEPlasma/sbatch_base_dp', single_run_timelim=81)
    runner.check_complete()
    # runner.clean()
    # runner.code_run(sampler.samples, run_id='test')


