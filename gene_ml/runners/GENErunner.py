import os
import sys
import subprocess
import numpy as np
from collections.abc import Iterator
import re
from ..tools import sec_to_time_format
import time
class GENErunner():
    def __init__(self, parser, config, remote_save_dir, time_model=None, single_run_timelim=None, single_run_simtimelim=None):
        self.parser=parser
        self.host = config.host
        self.single_run_timelim = single_run_timelim
        self.single_run_simtimelim = single_run_simtimelim
        self.remote_run_dir = config.remote_run_dir
        self.ssh_path = f"{self.host}:{self.remote_run_dir}"
        self.time_model = time_model
        #self.local_run_files_dir = config.local_run_files_dir
        self.max_wallseconds = 0
        self.config = config

        self.remote_save_dir = remote_save_dir

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

        print(f"CHECKING IF PROBLEM DIRECTORY EXISTS")
        problem_command = f'cd {self.remote_run_dir} && ./newprob && mv prob01 auto_prob_{run_id}; exit'
        if self.config.local:
            if not os.path.exists(remote_problem_dir):
                print("MAKING PROBLEM DIRECTORY")
                os.system(problem_command)
            else: print("PROBLEM DIRECTORY EXISTS")
        
        else:
            try:
                directory_details = self.config.paramiko_sftp_client.stat(remote_problem_dir)

            except FileNotFoundError:
                print('REMOTE PROBLEM DIRECTORY DOES NOT EXIST, CREATING IT NOW:', remote_problem_dir)
                _, stdout, _ = self.config.paramiko_ssh_client.exec_command(problem_command)
                print('RESULT FROM COMMAND:',stdout.read())

        self.parser.base_to_remote(remote_param_path, remote_sbatch_path)

        print(f"ALTERING THE PARAMETERS FILE IN THE REMOTE PROBLEM DIRECTORY")
        self.parser.alter_parameters_file(remote_param_path, group_var=["general","timelim"], value=self.single_run_timelim) # If using time_model this should be set with the time model in the sampler and put in the scan format on the parameters file.
        
        print('\n\nCODE RUN: SETTING SIMULATION TIME LIMMIT\n\n')
        # simtimelim is the timelimit inside the simulation, so number of seconds of plasma evolution. The simulation should be fater than walltime so I set it to the same time to ensure no limitations here.
        # self.parser.set_simtimelim(self.single_run_simtimelim, parameters_path=remote_param_path)
        self.parser.alter_parameters_file(remote_param_path, group_var=["general","simtimelim"], value=self.single_run_simtimelim) # If using time_model this should be set with the time model in the sampler and put in the scan format on the parameters file.
        
        print('SBATCH')
        print(self.parser.write_sbatch(remote_sbatch_path, remote_continue_path, wallseconds))

        print(f'PARSING SAMPLES TO INPUT FILE at:',remote_param_path)
        print(self.parser.write_input_file(samples, remote_param_path, remote_save_dir=os.path.join(self.remote_save_dir, run_id)))

    def code_run(self, samples, run_id):
        self.pre_run_check(samples, run_id)

        print('\nCODE RUN')
        # make results directory
        remote_save_dir_run_id = os.path.join(self.remote_save_dir, run_id)
        mkdir_command = f'mkdir -p {remote_save_dir_run_id}'

        print('MAKING PARSER REMOTE SAVE DIRECTORY,', remote_save_dir_run_id)
        
        remote_problem_dir = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}') 
        run_command = f'cd {self.remote_run_dir}/auto_prob_{run_id} && sbatch submit.cmd; exit'
        if self.config.local:
            mkdir_result = subprocess.run(mkdir_command, shell=True, capture_output=True, text=True)
            mkdir_out = mkdir_result.stdout
            mkdir_err = mkdir_result.stderr

            result = subprocess.run(run_command, shell=True, capture_output=True, text=True)
            out = result.stdout
            err = result.stderr
        else:
            mkdir_stdin, mkdir_stdout, mkdir_stderr = self.config.paramiko_ssh_client.exec_command(mkdir_command)
            mkdir_out = mkdir_stdout.read().decode('utf8')
            mkdir_err = mkdir_stderr.read().decode('utf8')

            stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(run_command)
            out = stdout.read().decode('utf8')
            err = stderr.read().decode('utf8')

        print('RESULT OF MAKING REMOTE SAVE DIRECTORY,', mkdir_out, mkdir_err)

        print('OUT:', out, 'ERROR?:', err)        
        sbatch_id = re.search('(?<![\d])\d{7}(?![\d])', out).group(0)
        print('OUT:', out, 'ERROR?:', err)
        print('SUBMITTED SBATCH ID:', sbatch_id)
        return sbatch_id
    
    # def kill_runs(self, sbatch_ids):
    #     print('\n KILLING SBATCH IDS',sbatch_ids)
    #     os.system(f"ssh {self.host} 'scancel {' '.join(sbatch_ids)}'")
    #     command = f"ssh {self.host} 'squeue --me'"
    #     queue = subprocess.check_output(command, shell=True, text=True)
    #     print('SLURM QUEUE AFTER KILL',queue)
    #     print('\n')

    def check_complete(self, run_ids):
        statuses = self.check_gene_status(run_ids)
        complete = []
        for status in statuses:
            if 's' in status:
                complete.append(False)
            else:
                complete.append(True)
        return(np.array(complete))
    
    def check_gene_status(self, run_ids):
        latest = self.get_latest_scanfiles_path(run_ids)
        status_paths = [os.path.join(l,'in_par','gene_status') for l in latest]
        status = []
        for status_path in status_paths:
            with self.parser.open_file(status_path, 'r') as status_file:
                status.append(str(status_file.read()))
        print('CHECK GENE STATUS:',status)
        return status
    

    # Checks to see if the slurm batch jobs are still in the queue. 
    def check_finished(self, sbatch_ids):
        print('\nCHECKING IF JOBS FINISHED:', sbatch_ids)
        #To check that certain sbatch_id's are no longer in the squeue
        command = f"squeue --me"
        if self.config.local:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            out = result.stdout
            err = result.stderr
        else:
            stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(command)
            out = stdout.read().decode('utf8')
        finished = np.array([not s_id in out for s_id in sbatch_ids])
        if all(finished):
            print('NONE OF THE INPUTED SBATCH IDs ARE RUNNING')
        else:
            print('FINISHED', finished)
            print('RUNNING SBATCH IDs: ', np.array(sbatch_ids)[~finished])
        print('\n')
        return all(finished)
    
    def wait_till_finished(self, sbatch_ids, check_interval=5):
        start = time.time()
        while not self.check_finished(sbatch_ids):
            time.sleep(check_interval)
            now = time.time()
            print(f'TIME SINCE STARTED  : {sec_to_time_format(now-start)}')
            print(f'MAX WALL TIME       :',sec_to_time_format(self.max_wallseconds))

    
    # def retrieve_run_file(self, file_name, run_id):
    #     print("GETTING RUN FILE: ", file_name)
    #     command = f"scp '{os.path.join(self.ssh_path, 'auto_prob_'+run_id, file_name)}' {os.path.join(self.local_run_files_dir, file_name)}"
    #     command = f"scp '{os.path.join(self.ssh_path, 'auto_prob_'+run_id, file_name)}' {os.path.join(self.local_run_files_dir, file_name)}"
        
    #     os.system(command)
    
    # Checks to see if the GENE run needs to be continued
    # def check_complete(self, run_ids):
    #     print('RUNNER CHECK COMPLETE')
    #     incomplete = []
    #     status = []
    #     for run_id in run_ids:
    #         scanfiles_dir = self.config.paramiko_sftp_client.listdir(os.path.join(self.parser.remote_save_dir,run_id))
    #         print('SCANFILES_DIR', scanfiles_dir)
    #         scanfiles_number = [re.findall('[0-9]{4}',sc_dir) for sc_dir in scanfiles_dir]
    #         print('SCANFILES_NUMBER', scanfiles_number)
    #         latest_scanfile = scanfiles_dir[np.argmax(np.array(scanfiles_number).astype('int'))]
    #         geneerr_log_path = os.path.join(self.parser.remote_save_dir,run_id,latest_scanfile,'geneerr.log')
    #         status.append(self.parser.hit_simtimelim_test(geneerr_log_path), get_status=True)
    #     incomplete = ['s' in s for s in status] #'s' stands for started. 'f' for finished, status is a string of s and f char one for each point in a run_id
    #     return incomplete, status
    #         #Trying to use new paramiko method for this.
            # out_path = os.path.join(self.parser.remote_save_dir, run_id)
            # command = f'ls {out_path}'
            # stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(command)
            # lines = stdout.readlines()
            # print('stdout',lines)
            # scan_numbers = [out[-4:] for out in lines]
            # print('SCAN NUMBERS',scan_numbers)
            # geneerr_log_path = os.path.join(self.parser.remote_save_dir, run_id)

            # regex_command = "ls | grep -E '[0-9]{7}.err$'"
            # command = f"ssh {self.host} 'cd {os.path.join(self.remote_run_dir,'auto_prob_'+run_id)} &&  {regex_command}; exit'"
            # files = subprocess.check_output(command, shell=True, text=True)
            # files = files.strip().split('\n')
            # ran_sbatch_ids = [f.strip().split('.')[-2] for f in files]
            # latest = files[np.argmax(ran_sbatch_ids)]
            # self.retrieve_run_file(latest, run_id)
            # scan_status = self.parser.read_scan_status(os.path.join(self.local_run_files_dir, latest))
            # if scan_status == 'needs continuation':
            #     incomplete.append(run_id)
        # return incomplete

    # Nothing after here has been adapted for local, potentially not paramiko either.    
    
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
        

    def continue_with_increased_omega_prec():
        None

    def continue_run(self, run_id):
        # if 
        print('CONTINUING RUN -', run_id)
        continue_command = f'cd {os.path.join(self.remote_run_dir,'auto_prob_'+run_id)} &&sbatch continue.cmd; exit'
        stdin, stdout, stderr = self.config.paramiko_ssh_client.exec_command(continue_command)
        out = stdout.read().decode('utf8')
        err = stderr.read().decode('utf8')
        print('OUT:',out, 'ERR:',err)
        batch_id = re.search('(?<![\d])\d{7}(?![\d])', out).group(0)
        print('SUBMITTED SBATCH ID:', batch_id)
        return batch_id    

    def continue_with_new_param(self, run_ids, group_var, value, perform_status_check=True):
        print(f'CONTINUING RUN {run_ids}\n with new {group_var}:{value}')
        #perform checks to see if we should increase to new_simtimelim
        if perform_status_check:
            print('PERFORMING STATUS CHECK')
            # Did all runs finish previously
            complete = self.check_complete(run_ids)
            restarts = 0
            if not all(complete):
                for i in range(restarts):
                    complete = self.check_complete(run_ids)
                    print('debug',complete, ~complete, np.array(run_ids)[~complete])
                    sbatch_ids = []
                    for run_id in np.array(run_ids)[~complete]:
                        sbatch_ids.append(self.continue_run(run_id))
                    self.wait_till_finished(sbatch_ids)
                if not all(complete):
                    raise RuntimeError(f"Daniel Says: These scans: \n{np.array(run_ids)[~complete]}\n didn't get to complete all the runs, even after {restarts}. This could be because the sbatch wall clock limit was reached or another error. Continue the scan untill they are all complete before tring to continue with a new parameter.")
        remote_prob_parameters = [os.path.join(self.remote_run_dir,'auto_prob_'+run_id, 'parameters') for run_id in run_ids]
        
        latest = self.get_latest_scanfiles_path(run_ids)
        remote_in_pars_dir = [os.path.join(l,'in_par') for l in latest]
        status_paths = [os.path.join(l,'in_par','gene_status') for l in latest]
        batch_ids = []

        for run_id, latest_i, rem_prob_param, remote_in_pars_dir, status_path in zip(run_ids, latest, remote_prob_parameters, remote_in_pars_dir, status_paths):
            
            # print('ALTERING THE PROB DIR PARAMETERS FILE:', rem_prob_param)
            # # put the new param value in the remote parameters file
            # self.parser.alter_parameters_file(rem_prob_param, group_var=group_var, value=value)
            # print('ALTERING THE in_par DIR PARAMETERS FILES')
            # # put the new param value in all remote parameters files in the inpar directory
            # for in_par_parameters in self.config.paramiko_sftp_client.listdir(remote_in_pars_dir):
            #     if 'parameters' in in_par_parameters:
            #         self.parser.alter_parameters_file(os.path.join(remote_in_pars_dir, in_par_parameters), group_var, value)
            # print('ALTERING THE OUT DIR PARAMETERS FILE')
            # self.parser.alter_parameters_file(os.path.join(latest_i, 'parameters'),group_var = group_var, value=value)

            self.alter_all_parameters_files(run_id, group_var, value, latest_i)

            print('ALTERING THE STATUS FILE TO CONTAIN ONLY s')
            # change the status file to a string of sssss so that all runs are started with new parameter value       
            with self.parser.open_file(status_path, 'r') as status_file:
                status = status_file.read().decode('utf8')
                print('STATUS FOUND', status)
            status = status.replace('f','s')
            with self.parser.open_file(status_path, 'w') as status_file:
                status_file.write(status)
            #double checking it is correctly altered
            with self.parser.open_file(status_path, 'r') as status_file:
                status = status_file.read().decode('utf8')
                print('NEW STATUS IN FILE', status)
            
            print('RUNNING continue.cmd IN THE PROBLEM DIRECTORY')
            batch_ids.append(self.continue_run(run_id))
        return batch_ids
    
    def alter_all_parameters_files(self, run_id, group_var, value, scanfile_dir):
        rem_prob_param = os.path.join(self.remote_run_dir,'auto_prob_'+run_id, 'parameters')
        remote_in_pars_dir = os.path.join(scanfile_dir,'in_par')
        print('ALTERING THE PROB DIR PARAMETERS FILE:', rem_prob_param)
        # put the new param value in the remote parameters file
        self.parser.alter_parameters_file(rem_prob_param, group_var=group_var, value=value)
        print('ALTERING THE in_par DIR PARAMETERS FILES')
        # put the new param value in all remote parameters files in the inpar directory
        for in_par_parameters in self.config.paramiko_sftp_client.listdir(remote_in_pars_dir):
            if 'parameters' in in_par_parameters:
                self.parser.alter_parameters_file(os.path.join(remote_in_pars_dir, in_par_parameters), group_var, value)
        print('ALTERING THE OUT DIR PARAMETERS FILE')
        self.parser.alter_parameters_file(os.path.join(scanfile_dir, 'parameters'),group_var = group_var, value=value)


    def continue_non_converged_runs(self, run_ids, new_simtimelim=None):
        # STILL TO BE TESTED
        #the main difference between this and continue_with_new_param is that geneerr.log will be checked to see which runs have not converged and so need to be continued. 
        latest = self.get_latest_scanfiles_path(run_ids)
        latest_geneerr_path = []
        max_stl_ids = []
        for latest_i in latest:
            files = self.config.paramiko_sftp_client.listdir(latest_i)
            max_stl_id = 0
            max_id_index = None
            for i, file in enumerate(files):
                if 'stl_id_' in file and 'geneerr.log' in file:
                    stl_id = int(re.search('id_(.*)_id', file).group(1))
                    if stl_id > max_stl_id: 
                        max_stl_id = stl_id
                        max_id_index = i
                elif 'geneerr.log' in file and type(max_id_index)==type(None):
                    max_id_index = i
            max_stl_ids.append(max_stl_id)
                
            latest_geneerr_path.append(os.path.join(latest_i, files[max_id_index]))
        new_stl_ids = [int(stl_id) + 1 for stl_id in max_stl_ids]
        print('PERFORMING CHECK_COMPLETE TO ENSURE ALL PREVIOUS RUNS FINISHED')
        complete = self.check_complete(run_ids=run_ids)
        if not all(complete):
            raise RuntimeError('Daniel Says: Something did not complete according to the gene_status file. This usually happends if the sbatch wall clock limit is too low, or a GENE error occured, please investigate.')
        
        print('LOOKING AT GENEERR.LOG TO DETERMIN A NEW STATUS SO THAT RUNS THAT HAVE NOT CONVERGED ARE CONTINUED')
        new_statuses = []
        for geneerr_path in latest_geneerr_path:
            status = self.parser.hit_simtimelim_test(geneerr_path, get_status=True)
            new_statuses.append(status)
        
        print('PLACING THE NEW STATUSES INTO THEIR PATHS')
        status_paths = [os.path.join(l,'in_par','gene_status') for l in latest]
        for status_path, new_status in zip(status_paths, new_statuses):
            self.parser.set_status(status_path, new_status)
        
        print('CHANGING THE NAME OF ANY IMPORTANT FILES SO THEY ARE NOT APPENDED OR OVERWRITTEN')
        for latest_i in latest:
            self.parser.rename_important_scanfiles(latest_i, prefix='old')

        if type(new_simtimelim) != type(None):
            print('ALTERING ALL PARAMETERS FILES')
            for run_id, latest_i in zip(run_ids, latest):
                self.alter_all_parameters_files(run_id, group_var=['general','simtimelim'], value=new_simtimelim, scanfile_dir=latest_i)
        
        print('CONTINUING WITH SBATCH CONTINUE.CMD IN THE PROBLEM DIRECTORIES')
        batch_ids = []
        for run_id in run_ids:
            batch_ids.append(self.continue_run(run_id))
        
        print('WAITING FOR CONTINUED RUNS TO FINISH SO THE FILES CAN BE RENAMED')
        self.wait_till_finished(batch_ids, check_interval=10)

        print('CHANGING THE NAME OF THE MADE FILES SO THEY CAN BE IDENTIFIED LATER')
        for latest_i, stl_id in zip(latest, new_stl_ids):
            if type(new_simtimelim) != type(None):            
                self.parser.rename_important_scanfiles(latest_i, prefix=f'stl_{new_simtimelim}_id_{stl_id}_id')
                print(f'FINISHED CONTINUE NUMBER {stl_id}, IN SCANFILES {latest_i} WITH NEW SIMTIMELIM:', new_simtimelim)
            else:
                self.parser.rename_important_scanfiles(latest_i, prefix=f'stl_id_{stl_id}_id')
                print(f'FINISHED CONTINUE NUMBER {stl_id}, IN SCANFILES {latest_i}')
        

        
        


    # def continue_with_increased_simtimelim(self, run_ids, new_simtimelim):
    #     print('RUNNER, continue with increased simtimelim\n')
    #     remote_parameters_paths = [os.path.join(self.remote_run_dir,'auto_prob_'+run_id,'parameters') for run_id in run_ids]

    #     latest = self.get_latest_scanfiles_path(run_ids)
    #     geneerr_paths = [os.path.join(l,'geneerr.log') for l in latest]
    #     status_paths = [os.path.join(l,'in_par','gene_status') for l in latest]
    #     batch_ids = []
    #     for run_id, rem_param_path, gerr_path, status_path in zip(run_ids, remote_parameters_paths, geneerr_paths, status_paths):
    #         #perform checks to see if we should increase to new_simtimelim
    #         print('PERFORMING CHECKS TO SEE IF SIMTIMELIM SHOULD BE INCREASED')
    #         status = self.parser.hit_simtimelim_test(gerr_path, get_status=True)
    #         if not 's' in status:
    #             print(f"NO GENE RUNS HIT THE SIM TIME LIMIT SO NOT INCREASING AND NOT CONTINUING THE RUN,{run_id}")

    #         old_simtimelim = self.parser.get_parameter_value(rem_param_path, group_var=['general','simtimelim'])
    #         print('old simtime lim', old_simtimelim)
    #         if old_simtimelim >= new_simtimelim:
    #             print('simtime lime old', old_simtimelim, 'new',new_simtimelim)
    #             raise ValueError('Daniel Says: You tried to increase the simtimelim but the value entered is the same or smaller than what was already there. Please try again with a larger number.')
            
    #         print(f'CONTINUING RUN {run_id} with new simtimelim:{new_simtimelim}')
    #         # put the new sim time limit in the remote parameters file
    #         self.parser.alter_parameters_file(rem_param_path, group_var=['general','simtimelim'], value=new_simtimelim)
    #         # alter the status file so the continue scan will know what to run and what to leave finished.
    #         with self.parser.open_file(status_path, 'w') as status_file:
    #             status_file.write(status)
            
    #         batch_ids.append(self.continue_run(run_id))
    #     return batch_ids
            
    def get_latest_scanfiles_path(self, run_ids):
        latest = []
        for run_id in run_ids:
            remote_save = os.path.join(self.remote_save_dir, run_id)
            scanfiles_dir = self.config.paramiko_sftp_client.listdir(remote_save)
            scanfiles_number = [re.findall('[0-9]{4}',sc_dir) for sc_dir in scanfiles_dir]
            latest_scanfile = scanfiles_dir[np.argmax(np.array(scanfiles_number).astype('int'))]
            latest.append(os.path.join(remote_save,latest_scanfile))
        return latest

    # def clean(self):
    #     '''
    #         Removes any prob0* directories created by ./newprob and the auto_prob_* directories created for the specific run.
    #     '''
    #     print('CLEANING RUN DIR OF RUNER CREATED DIRECTORIES')
    #     os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && rm -r auto_prob_* prob0*'")

    #     os.system('rm temp/*')

        

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


