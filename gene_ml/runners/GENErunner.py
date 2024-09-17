import os
import sys
import subprocess
import numpy as np
from collections.abc import Iterator
class GENErunner():
    def __init__(self, parser, host, sbatch_base_path, remote_run_dir, time_model=None, guess_sample_wallseconds=None, local_run_files_dir=os.path.join(os.getcwd(),'run_files')):
        self.parser=parser
        self.host = host
        self.sbatch_base_path = sbatch_base_path
        self.guess_sample_wallseconds = guess_sample_wallseconds
        self.remote_run_dir = remote_run_dir
        self.ssh_path = f"{self.host}:{self.remote_run_dir}"
        self.time_model = time_model
        self.local_run_files_dir = local_run_files_dir
        self.max_wallseconds = 0

    def sec_to_time_format(self, sec):
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60, )
            d, h = divmod(h, 24)
            s,m,h,d = str(int(s)), str(int(m)), str(int(h)), str(int(d))
            if len(d)==1: d = '0'+d
            if len(h)==1: h = '0'+h
            if len(m)==1: m = '0'+m
            if len(d)==1: s = '0'+s
            return f"{d}-{h}:{m}:{s}"

    def generate_sbatch(self, wallseconds , run_id):

        sbatch = open(self.sbatch_base_path, "r").read()
        # parameters_scan = open(parameters_path, "r").read()

        # first_scanwith_loc = parameters_scan.find('!scanwith:')
        # n_samples = len(parameters_scan[first_scanwith_loc:parameters_scan.find('\n', first_scanwith_loc)].split(','))-1
        wall_clock_limit = self.sec_to_time_format(wallseconds)
        print(f'WALL CLOCK LIMIT FOR BATCH {run_id}:  ', wall_clock_limit)
        sbatch_lines = sbatch.split('\n')
        wall_loc = 0
        for i in range(len(sbatch_lines)):
            if '#SBATCH -t' in sbatch_lines[i]: 
                wall_loc = i
                break
        sbatch_lines[wall_loc] = f"#SBATCH -t {wall_clock_limit}  # wallremote_run_dir = '/project/project_462000451/gene/'clock limit, dd-hh:mm:ss"

        sbatch = "\n".join(sbatch_lines)
        if not os.path.exists('temp/'): os.mkdir('temp/')
        with open(f'temp/sbatch_{run_id}', "w") as sbatch_file:
            sbatch_file.write(sbatch)

        for i, line in enumerate(sbatch_lines):
            if "./scanscript" in line:
                sbatch_lines[i] = line.replace("./scanscript", "./scanscript --continue_scan")

        continue_str = "\n".join(sbatch_lines)
        with open(f'temp/continue_{run_id}', "w") as continue_file:
            continue_file.write(continue_str)
        

        
        #Important question, what affects how long a gene simulation takes
        #       Is it just paralellisation and numerical parameters or do the physicsal parameteters also matter.
        #       in the scanfiles there are two files called geneerr.log and geneerr.log_0001_eff.
        #       Both seem the same and have a total wallclock time at the end. Can you help me decipher which time is the total wall time for all the samples to run
    
    
    def continue_run(self, run_id):
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

                

    def code_run(self, samples, run_id):
        print('\nCODE RUN')
        # make results directory
        error_code = os.system(f"ssh {self.host} 'mkdir -p {self.parser.remote_save_dir}'")
        if error_code != 0: 
            raise SystemError('When making the results directory with ssh there was an error')

        if not os.path.exists('temp/'):
            os.mkdir('temp/')

        #using time model or guess sample walltime to get the walltime
        if type(self.time_model)!=type(None):
            times, errors = self.time_model.predict(samples)
            wallseconds = np.sum(times) * 1.30
        else:
            n_samples = len(list(samples.values())[0])
            wallseconds = self.guess_sample_wallseconds * n_samples * 1.30 #add 30% more to ensure it works
        if wallseconds > self.max_wallseconds: self.max_wallseconds = wallseconds
        print(f"THE ESTIMATED WALLTIME FOR RUN {run_id} is {self.sec_to_time_format(wallseconds)}, dd-hh-mm-ss TO RUN {n_samples} SAMPLES")

        print(f"ALTERING THE BASE PARAMETERS FILE TO SET THE TIMELIM AND SIMTIMELIM TO THE WALLTIME")
        self.parser.alter_base(group_var="general_timelim", value=int(wallseconds))
        # simtimelim is the timelimit inside the simulation, so number of seconds of plasma evolution. The simulation should be fater than walltime so I set it to the same time to ensure no limitations here.
        self.parser.alter_base(group_var="general_simtimelim", value=int(wallseconds))

        print(f'PARSING SAMPLES TO INPUT FILE at temp/parameters_{run_id}')
        self.parser.write_input_file(samples, file_name=f'parameters_{run_id}')
        print(f'GENERATING SBATCH FROM PARAMETERS FILE at temp/sbatch_{run_id}')
        
        self.generate_sbatch(wallseconds,run_id=run_id)
        print('CREATING A NEW PROBLEM DIR WITH SSH')
        remote_param_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'parameters')
        remote_sbatch_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'submit.cmd')
        remote_continue_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'continue.cmd')
        print('MOVING PARAMETERS AND SBATCH FILES TO CORRECT LOCATION IN REMOTE; SUBMITTING GENE SBATCH')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && ./newprob && mv prob01 auto_prob_{run_id}; exit' && scp temp/parameters_{run_id} {self.host}:{remote_param_path} && scp temp/sbatch_{run_id} {self.host}:{remote_sbatch_path} && scp temp/continue_{run_id} {self.host}:{remote_continue_path}")

        print(f'CREATING auto_prob_{run_id} in {self.remote_run_dir}') 
        command = f"ssh {self.host} 'cd {self.remote_run_dir}/auto_prob_{run_id}; sbatch submit.cmd; exit'"
        sbatch_id = subprocess.check_output(command, shell=True, text=True)
        sbatch_id = sbatch_id.strip().split(' ')[-1]
        
        print('SUBMITTED SBATCH ID',sbatch_id)
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
            regex_command = "ls | grep -E '[0-9]{7}.err$'"
            command = f"ssh {self.host} 'cd {os.path.join(self.remote_run_dir,'auto_prob_'+run_id)} &&  {regex_command}; exit'"
            files = subprocess.check_output(command, shell=True, text=True)
            files = files.strip().split('\n')
            ran_sbatch_ids = [f.strip().split('.')[-2] for f in files]
            latest = files[np.argmax(ran_sbatch_ids)]
            self.retrieve_run_file(latest, run_id)
            scan_status = self.parser.read_scan_status(os.path.join(self.local_run_files_dir, latest))
            if scan_status == 'ready for continuation':
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
    runner = GENErunner(parser, remote_run_dir='/project/project_462000451/gene_auto/', host='lumi', sbatch_base_path = '/home/djdaniel/DEEPlasma/sbatch_base_dp', guess_sample_wallseconds=81)
    runner.check_complete()
    # runner.clean()
    # runner.code_run(sampler.samples, run_id='test')


