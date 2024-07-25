import os
import sys
import subprocess
import numpy as np
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

    def generate_sbatch(self, wallseconds , run_id):
        def sec_to_time_format(sec):
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60, )
            d, h = divmod(h, 24)
            s,m,h,d = str(int(s)), str(int(m)), str(int(h)), str(int(d))
            if len(d)==1: d = '0'+d
            if len(h)==1: h = '0'+h
            if len(m)==1: m = '0'+m
            if len(d)==1: s = '0'+s
            return f"{d}-{h}:{m}:{s}"

        sbatch = open(self.sbatch_base_path, "r").read()
        # parameters_scan = open(parameters_path, "r").read()

        # first_scanwith_loc = parameters_scan.find('!scanwith:')
        # n_samples = len(parameters_scan[first_scanwith_loc:parameters_scan.find('\n', first_scanwith_loc)].split(','))-1
        wall_clock_limit = sec_to_time_format(wallseconds)
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
        
        
        #Important question, what affects how long a gene simulation takes
        #       Is it just paralellisation and numerical parameters or do the physicsal parameteters also matter.
        #       in the scanfiles there are two files called geneerr.log and geneerr.log_0001_eff.
        #       Both seem the same and have a total wallclock time at the end. Can you help me decipher which time is the total wall time for all the samples to run


    def code_run(self, samples, run_id):
        # make results directory
        error_code = os.system(f"ssh {self.host} 'mkdir -p {self.parser.remote_save_dir}'")
        if error_code != 0: 
            print('When making the results directory with ssh there was an error')
            raise SystemError

        if not os.path.exists('temp/'):
            os.mkdir('temp/')
        # os.system('rm temp/*')

        #the simulation wall time limit needs to be slightly smaller than the wall time to ensure enough time for checkpoints. So the wall time is set to be 30% greater than guessed time
        if type(self.time_model)!=type(None):
            times, errors = self.time_model.predict(samples)
            wallseconds = np.sum(times) * 1.30
        else:
            n_samples = len(list(samples.values())[0])
            wallseconds = self.guess_sample_wallseconds * n_samples * 1.30 #add 30% more to ensure it works
        
        self.parser.alter_base(group_var="general_timelim", value=wallseconds)
        # simtimelim is the timelimit inside the simulation, so number of seconds of plasma evolution. The simulation should be fater than walltime so I set it to the same time to ensure no limitations here.
        self.parser.alter_base(group_var="general_simtimelim", value=wallseconds)

        print(f'PARSING SAMPLES TO INPUT FILE at temp/parameters_{run_id}')
        self.parser.write_input_file(samples, file_name=f'parameters_{run_id}')
        print(f'GENERATING SBATCH FROM PARAMETERS FILE at temp/sbatch_{run_id}')
        
        self.generate_sbatch(wallseconds,run_id=run_id)
        print('CREATING A NEW PROBLEM DIR WITH SSH')
        remote_param_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'parameters')
        remote_sbatch_path = os.path.join(self.remote_run_dir, f'auto_prob_{run_id}', 'submit.cmd')
        print('MOVING PARAMETERS AND SBATCH FILES TO CORRECT LOCATION IN REMOTE; SUBMITTING GENE SBATCH')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && ./newprob && mv prob01 auto_prob_{run_id}; exit' && scp temp/parameters_{run_id} {self.host}:{remote_param_path} && scp temp/sbatch_{run_id} {self.host}:{remote_sbatch_path}")
         
        command = f"ssh {self.host} 'cd {self.remote_run_dir}/auto_prob_{run_id}; sbatch submit.cmd; exit'"
        sbatch_code = subprocess.check_output(command, shell=True, text=True)
        print(sbatch_code.strip().split(' ')[-1])
        
        print('\n\nSBATCH CODE',sbatch_code)
        print(f'JUST CREATED auto_prob_{run_id} in remote_run_dir')
        return sbatch_code

    def check_finished(self, sbatch_ids):
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
        return finished
    
    def retrieve_run_file(self, file_name, run_id):
        print("GETTING RUN FILE: ", file_name)
        command = f"scp '{os.path.join(self.ssh_path, 'auto_prob_'+run_id, file_name)}' {os.path.join(self.local_run_files_dir, file_name)}"
        os.system(command)

        
    def check_complete(self, run_ids):
        for run_id in run_ids:
            regex_command = "ls | grep -E '[0-9]{7}\.out$'"
            command = f"ssh {self.host} 'cd {os.path.join(self.remote_run_dir,'auto_prob_'+run_id)} &&  {regex_command}; exit'"
            files = subprocess.check_output(command, shell=True, text=True)
            files = files.strip().split('\n')
            ran_sbatch_ids = [f.strip().split('.')[-2] for f in files]
            latest = files[np.argmax(ran_sbatch_ids)]
            self.retrieve_run_file(latest, run_id)
            self.parser.read_scan_status(os.path.join(self.local_run_files_dir, latest))




            # print(files)
            

        
        
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


