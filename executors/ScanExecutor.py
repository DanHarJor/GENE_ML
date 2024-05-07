import numpy as np
import sys
sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/')    

try:
    from .base import Executor
except:
    try:
        from base import Executor
    except:
        raise ImportError

class ScanExecutor():
    def __init__(self, sbatch_base_path, num_workers, sampler, runner, **kwargs):
        # super().__init__(**kwargs)
        self.num_workers = num_workers  # kwargs.get('num_workers')
        self.sampler = sampler
        self.runner = runner
        self.sbatch_base_path = sbatch_base_path
        print('Beginning Batch Script Generation')

    def generate_sub_sbatch():
        raise NotImplemented

    def start_runs(self):
        #batches is a list of dictionaries where each one is a subset of the entire samples dictionary.
        batches = {k:np.split(v,self.num_workers) for k,v in self.sampler.samples.items()}
        batches = [{k:v[i] for k,v in batches.items()} for i in range(self.num_workers)]
        print(100 * "=")
        print("Starting Dataset generation")
        print("Creating initial runs")

        for batch, id in zip(batches, np.arange(len(batches))):
            self.runner.code_run(batch, sbatch_path=self.sbatch_base_path, remote_gene_dir='/project/project_462000451/gene_auto/', host_name='lumi', id=id)

if __name__ == '__main__':
    import os
    import sys
    sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/')
    from samplers.uniform import Uniform
    from runners.GENErunner import GENErunner
    sys.path.append(os.path.join('/home/djdaniel/DEEPlasma','GENE_ML','enchanted-surrogates','src'))
    from parsers.GENEparser import GENE_scan_parser
    parameters = ['box-kymin', '_grp_species_1-omt', 'species-omn']
    bounds = [(0.05,1), (10,70), (5,60)]
    num_samples = 10
    sampler = Uniform(parameters=parameters, bounds=bounds, num_samples=num_samples)
    base_params_path = os.path.join(os.getcwd(),'parameters_base_dp')
    parser = GENE_scan_parser(base_params_path)
    runner = GENErunner(parser)
    executor = ScanExecutor(num_workers=5, sbatch_base_path='/home/djdaniel/DEEPlasma/sbatch_base_dp' ,sampler=sampler, runner=runner, base_run_dir=None)
    executor.start_runs()
