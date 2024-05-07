import os
import sys
import subprocess
class GENErunner():
    def __init__(self, parser):
        self.parser=parser
    
    def code_run(self, samples, sbatch_path, remote_gene_dir, host_name, id):
        if not os.path.exists('temp/'):
            os.mkdir('temp/')

        print('PARSING SAMPLES TO INPUT FILE')
        self.parser.write_input_file(samples, run_dir='temp/', file_name=f'parameters_{id}')
        print('CREATING A NEW PROBLEM DIR WITH SSH')
        remote_param_path = os.path.join(remote_gene_dir, f'prob_{id}', 'parameters')
        remote_sbatch_path = os.path.join(remote_gene_dir, f'prob_{id}', 'submit.cmd')
        os.system(f"ssh {host_name} 'cd {remote_gene_dir} && rm -r prob_{id} prob0* ; ./newprob && mv prob01 prob_{id}' && scp temp/parameters_{id} {host_name}:{remote_param_path} && scp {sbatch_path} {host_name}:{remote_sbatch_path} && ssh {host_name} 'cd {remote_gene_dir}/prob_{id}; sbatch submit.cmd'")
        print('MOVING PARAMETERS AND SBATCH FILES TO CORRECT LOCATION IN REMOTE')
        print('RUNNING GENE')
        # #clean up
        # os.system("rm /home/djdaniel/DEEPlasma/GENE_ML/temp/*")
        # os.system(f"ssh {host_name} 'cd {remote_gene_dir}; rm -r prob*'")

        

if __name__ == '__main__':
    import os
    sys.path.append(os.path.join('/home/djdaniel/DEEPlasma'))
    from GENE_ML.samplers.uniform import Uniform
    sys.path.append(os.path.join('/home/djdaniel/DEEPlasma','GENE_ML','enchanted-surrogates','src'))
    from parsers.GENEparser import GENE_scan_parser
    
    parameters = ['box-kymin', '_grp_species_1-omt', 'species-omn']
    bounds = [(0.05,1), (10,70), (5,60)]
    num_samples = 3
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    base_params_path = os.path.join(os.getcwd(),'parameters_base_dp')
    parser = GENE_scan_parser(base_params_path)
    runner = GENErunner(parser)
    runner.code_run(sampler.samples, sbatch_path='/home/djdaniel/DEEPlasma/sbatch_base_dp', remote_gene_dir='/project/project_462000451/gene_auto/', host_name='lumi', id='test')


