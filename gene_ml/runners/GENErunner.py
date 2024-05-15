import os
import sys
import subprocess
import numpy as np
class GENErunner():
    def __init__(self, parser, remote_run_dir, host, sbatch_base_path, guess_sample_wallseconds):
        self.parser=parser
        self.remote_run_dir = remote_run_dir
        self.host = host
        self.sbatch_base_path = sbatch_base_path
        self.guess_sample_wallseconds = guess_sample_wallseconds

    def generate_sbatch(self, parameters_path, id):
                
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
        parameters_scan = open(parameters_path, "r").read()

        first_scanwith_loc = parameters_scan.find('!scanwith:')
        n_samples = len(parameters_scan[first_scanwith_loc:parameters_scan.find('\n', first_scanwith_loc)].split(','))-1
        
        wallseconds = self.guess_sample_wallseconds * n_samples * 1.30 #add 30% more to ensure it works
        wall_clock_limit = sec_to_time_format(wallseconds)
        sbatch_lines = sbatch.split('\n')
        wall_loc = 0
        for i in range(len(sbatch_lines)):
            if '#SBATCH -t' in sbatch_lines[i]: 
                wall_loc = i
                break
        sbatch_lines[wall_loc] = f"#SBATCH -t {wall_clock_limit}  # wallclock limit, dd:hh:mm:ss"

        sbatch = "\n".join(sbatch_lines)
        if not os.path.exists('temp/'): os.mkdir('temp/')
        with open(f'temp/sbatch_{id}', "w") as sbatch_file:
            sbatch_file.write(sbatch)
        
        
        #Important question, what affects how long a gene simulation takes
        #       Is it just paralellisation and numerical parameters or do the physicsal parameteters also matter.
        #       in the scanfiles there are two files called geneerr.log and geneerr.log_0001_eff.
        #       Both seem the same and have a total wallclock time at the end. Can you help me decipher which time is the total wall time for all the samples to run


    def code_run(self, samples, id):
        if not os.path.exists('temp/'):
            os.mkdir('temp/')
        os.system('rm temp/*')
        print(f'PARSING SAMPLES TO INPUT FILE at temp/parameters_{id}')
        self.parser.write_input_file(samples, run_dir='temp/', file_name=f'parameters_{id}')
        print(f'GENERATING SBATCH FROM PARAMETERS FILE at temp/sbatch_{id}')
        self.generate_sbatch(os.path.join('temp',f'parameters_{id}'),id=id)
        print('CREATING A NEW PROBLEM DIR WITH SSH')
        remote_param_path = os.path.join(self.remote_run_dir, f'prob_{id}', 'parameters')
        remote_sbatch_path = os.path.join(self.remote_run_dir, f'prob_{id}', 'submit.cmd')
        print('MOVING PARAMETERS AND SBATCH FILES TO CORRECT LOCATION IN REMOTE; SUBMITTING GENE SBATCH')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && ./newprob && mv prob01 prob_{id}; exit' && scp temp/parameters_{id} {self.host}:{remote_param_path} && scp temp/sbatch_{id} {self.host}:{remote_sbatch_path} && ssh {self.host} 'cd {self.remote_run_dir}/prob_{id}; sbatch submit.cmd; exit'")
        
    def clean(self):
        '''
            Removes any prob0* directories created by ./newprob and the prob_* directories created for the specific run.
        '''
        print('CLEANING RUN DIR OF RUNER CREATED DIRECTORIES')
        os.system(f"ssh {self.host} 'cd {self.remote_run_dir} && rm -r prob_* prob0*'")
        

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
    parser = GENE_scan_parser(base_params_path)
    runner = GENErunner(parser, remote_run_dir='/project/project_462000451/gene_auto/', host='lumi', sbatch_base_path = '/home/djdaniel/DEEPlasma/sbatch_base_dp', guess_sample_wallseconds=81)
    runner.clean()
    runner.code_run(sampler.samples, id='test')


