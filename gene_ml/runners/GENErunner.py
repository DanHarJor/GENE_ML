import os
import sys
import subprocess
import numpy as np
class GENErunner():
    def __init__(self, parser, host, sbatch_base_path, guess_sample_wallseconds, remote_run_dir):
        self.parser=parser
        self.host = host
        self.sbatch_base_path = sbatch_base_path
        self.guess_sample_wallseconds = guess_sample_wallseconds
        self.remote_run_dir = remote_run_dir

    def generate_sbatch(self, n_samples, id):
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
        
        wallseconds = self.guess_sample_wallseconds * n_samples * 1.30 #add 30% more to ensure it works
        wall_clock_limit = sec_to_time_format(wallseconds)
        print(f'WALL CLOCK LIMIT FOR BATCH {id}:  ', wall_clock_limit)
        sbatch_lines = sbatch.split('\n')
        wall_loc = 0
        for i in range(len(sbatch_lines)):
            if '#SBATCH -t' in sbatch_lines[i]: 
                wall_loc = i
                break
        sbatch_lines[wall_loc] = f"#SBATCH -t {wall_clock_limit}  # wallremote_run_dir = '/project/project_462000451/gene/'clock limit, dd-hh:mm:ss"

        sbatch = "\n".join(sbatch_lines)
        if not os.path.exists('temp/'): os.mkdir('temp/')
        with open(f'temp/sbatch_{id}', "w") as sbatch_file:
            sbatch_file.write(sbatch)
        
        
        #Important question, what affects how long a gene simulation takes
        #       Is it just paralellisation and numerical parameters or do the physicsal parameteters also matter.
        #       in the scanfiles there are two files called geneerr.log and geneerr.log_0001_eff.
        #       Both seem the same and have a total wallclock time at the end. Can you help me decipher which time is the total wall time for all the samples to run


    def code_run(self, samples, id):
        # make results directory
        error_code = os.system(f"ssh {self.host} 'mkdir -p {self.parser.remote_save_dir}'")
        if error_code != 0: 
            print('When making the results directory with ssh there was an error')
            raise SystemError

        if not os.path.exists('temp/'):
            os.mkdir('temp/')
        os.system('rm temp/*')

        n_samples = len(list(samples.values())[0])
        #the simulation wall time limit needs to be slightly smaller than the wall time to ensure enough time for checkpoints. So the wall time is set to be 30% greater than guessed time
        timelim = self.guess_sample_wallseconds * n_samples
        self.parser.alter_base(group_var="general_timelim", value=timelim)
        # simtimelim is the timelimit inside the simulation, so number of seconds of plasma evolution. The simulation should be fater than walltime so I set it to the same time to ensure no limitations here.
        self.parser.alter_base(group_var="general_simtimelim", value=timelim)

        print(f'PARSING SAMPLES TO INPUT FILE at temp/parameters_{id}')
        self.parser.write_input_file(samples, file_name=f'parameters_{id}')
        print(f'GENERATING SBATCH FROM PARAMETERS FILE at temp/sbatch_{id}')
        self.generate_sbatch(n_samples,id=id)
        print('CREATING A NEW PROBLEM DIR WITH SSH')
        remote_param_path = os.path.join(self.remote_run_dir, f'auto_prob_{id}', 'parameters')
        remote_sbatch_path = os.path.join(self.remote_run_dir, f'auto_prob_{id}', 'submit.cmd')
        print('MOVING PARAMETERS AND SBATCH FILES TO CORRECT LOCATION IN REMOTE; SUBMITTING GENE SBATCH')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && ./newprob && mv prob01 auto_prob_{id}; exit' && scp temp/parameters_{id} {self.host}:{remote_param_path} && scp temp/sbatch_{id} {self.host}:{remote_sbatch_path} && ssh {self.host} 'cd {self.remote_run_dir}/auto_prob_{id}; sbatch submit.cmd; exit'")
    

    def clean(self):
        '''
            Removes any prob0* directories created by ./newprob and the auto_prob_* directories created for the specific run.
        '''
        print('CLEANING RUN DIR OF RUNER CREATED DIRECTORIES')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && rm -r auto_prob_* prob0*'")
        

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
    runner.clean()
    runner.code_run(sampler.samples, id='test')


