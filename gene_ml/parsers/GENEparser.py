import subprocess
import os
import f90nml.parser
import numpy as np
import f90nml
from typing import List
from copy import deepcopy
import pandas as pd
import re


# class GENEparser():
#     """An I/O parser for GENE

#      Attributes
#     ----------

#     Methods
#     -------
#     write_input_file
#         Writes the inputfile for a single set of parameters.

#     read_output_file
#         Reads the output file to python format

#     """
#     def __init__(self, base_params_dir=None, remote_save_dir=None):
#         """
#         Generates the base f90nml namelist from the GENE parameters file at base_params_dir.

#         Parameters
#         ----------
#             base_params_dir (string or path): The directory pointing to the base GENE parameters file.
#             The base GENE parameters file must contain all parameters necessary for GENE to run.
#             Any parameters to be sampled will be inserted into the base parameter file before each run.
#             Any value of a sampled parameter in the base file will be ignored. 
#         Returns
#         -------
#             Nothing 
#         """
#         if base_params_dir!=None:
#             self.base_namelist = f90nml.read(base_params_dir) #odict_keys(['parallelization', 'box', 'in_out', 'general', 'geometry', '_grp_species_0', '_grp_species_1', 'units'])
#         if remote_save_dir != None:
#             self.remote_save_dir = remote_save_dir

class GENE_single_parser():

    def write_input_file(self, params: dict, save_dir):
        """
        Write the GENE input file to the run directory specified. 
        
        Parameters
        ----------
            params (dict): The keys store strings of the names of the parameters as specified in the enchanted surrogates *_config.yaml configuration file.
            The values stores floats of the parameter values to be ran in GENE.

            rprint('Writing to', save_dir)
        if os.path.exists(save_dir):
            input_fpath = os.path.join(save_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {save_dir}')un_dir (string or path): The file system directory where runs are to be stored

        """
        print('Writing to', save_dir)
        if os.path.exists(save_dir):
            self.input_fpath = os.path.join(save_dir, 'parameters')
        else:
            raise FileNotFoundError(f'Couldnt find {save_dir}')

        params_keys = list(params.keys())
        params_values = list(params.values())
        patch = {}
        patch['in_out'] = {'diagdir':self.remote_save_dir}
        for key, value in zip(params_keys,params_values):
            group_name, variable_name = key.split('-')
            if list(patch.keys()).count(group_name) > 0:
                patch[group_name][variable_name] = value
            else: patch[group_name] = {variable_name:value}


        namelist = self.base_namelist
        patch = f90nml.namelist.Namelist(patch)
        namelist.patch(patch)
        
        f90nml.write(namelist, self.input_fpath)

    # what is returned here is returned to the runner for a single code run, which goes though the base executor to get to the future 
    def read_output_file(self, save_dir: str):
        raise NotImplementedError
    
class GENE_scan_parser(): 
    def __init__(self, save_dir, base_params_dir=None, remote_save_dir=None):
        """
        Generates the base f90nml namelist from the GENE parameters file at base_params_dir.

        Parameters
        ----------
            base_params_dir (string or path): The directory pointing to the base GENE parameters file.
            The base GENE parameters file must contain all parameters necessary for GENE to run.
            Any parameters to be sampled will be inserted into the base parameter file before each run.
            Any value of a sampled parameter in the base file will be ignored. 
        Returns
        -------
            Nothing 
        """
        self.save_dir = save_dir
        self.base_params_dir = base_params_dir
        if base_params_dir!=None:
            self.base_namelist = f90nml.read(self.base_params_dir) #odict_keys(['parallelization', 'box', 'in_out', 'general', 'geometry', '_grp_species_0', '_grp_species_1', 'units'])
        if remote_save_dir != None:
            self.remote_save_dir = remote_save_dir

    def alter_base(self, group_var, value):
        #currently only works for variables that only appear in one group, not omn as it is in each species group
        #var example var="general_timelim", group_variable for fortran parameters file.
        group, var = group_var.split("_")
        patch = {group: {var: value}}

        self.base_namelist.patch(patch)

        print('Writing to', self.base_params_dir)
        if os.path.exists(self.base_params_dir):
            f90nml.write(self.base_namelist, self.base_params_dir, force=True)
        else:
            raise FileNotFoundError(f'Couldnt find {self.base_params_dir}')

    #puts in the paramaters with the GENE !scan functionality
    def write_input_file(self, params: dict, file_name='parameters'):
        namelist = self.base_namelist
        namelist_string=str(namelist)
        
        #populate params: dict with all omn's required. Since each should be identical
        if 'species-omn' in params:
            for i in range(namelist_string.count('&species')):
                params[f'_grp_species_{i}-omn'] = params['species-omn']
            params.pop('species-omn')
        
        def find_nth_occurrence(string, sub_string, n):
            start_index = string.find(sub_string)
            while start_index >= 0 and n > 1:
                start_index = string.find(sub_string, start_index + 1)
                n -= 1
            return start_index

        # finds the string location at the end of the line for a variable, just before \n
        def var_end_loc(namelist_string: str, param_key):
            group_name, var_name = param_key.split('-')
            # print('GROUP NAME',group_name,'VAR NAME',var_name)
            group_ordinal = 0 #0 is the 1st
            if len(group_name.split('_'))>1:
                # print('MORE THAN ONE GROUP OF SAME NAME')
                _, _, group_name, group_ordinal = group_name.split('_')
                group_ordinal = int(group_ordinal)+1

            # print('GROUP NAME',group_name,'VAR NAME',var_name, 'ORDIANL',group_ordinal)

            group_start = find_nth_occurrence(namelist_string, group_name, group_ordinal)
            group_end = group_start+namelist_string[group_start:].find(f'/')
            # print('GROUP',namelist_string[group_start:group_end])

            var_start = group_start+namelist_string[group_start:group_end].find(var_name)
            var_end = var_start+namelist_string[var_start:group_end].find("\n")
            # print('VARLOC',var_start,var_end)
            # print('VAR',namelist_string[var_start:var_end])
            # print('START',namelist_string[var_start],'END',namelist_string[var_end])
            return var_end

        def make_scanlist(values):
        # Making scanlist
            scanlist = f'      !scanlist: {values[0]}'
            for v in values[1:]:
                scanlist += f', {v}'
            return scanlist
        #----------------------
        
        #determines the ordinal position of the scanned paramters for var_name
        def var_ordinal(param_key):
            group, var_name = param_key.split('-')
            group_split = group.split('_')
            if len(group_split)>1:
                group_name = group_split[-2]
                group_ord = int(group_split[-1])+1
            else:
                group_name = group
                group_ord = 1

            gloc = find_nth_occurrence(namelist_string, group_name, group_ord)
            vloc = gloc + namelist_string[gloc:].find(var_name)
            var_ordinal = namelist_string[:vloc].count('=')+1
            return var_ordinal
        
        #check which parameter is the first to be scanned and make is a scanlist
        ordinals = {k:var_ordinal(k) for k in params.keys()}
        first_param = None
        first_param_ord = np.inf
        for k,ord in ordinals.items():
            if ord < first_param_ord: 
                first_param_ord = ord
                first_param = k
        
        def make_scanwith(values):
            scanwith = f'       !scanwith: 0, {values[0]}'
            for v in values[1:]:
                scanwith += f', {v}'
            return scanwith

        scanwith = {k:make_scanwith(values) for k,values in params.items()}
        # Add scanwith to each variable with and scanlist to the first one
        for param_key in list(params.keys()):
            if param_key == first_param:
                var_end = var_end_loc(namelist_string,param_key)
                namelist_string = namelist_string[:var_end] + make_scanlist(params[param_key]) + namelist_string[var_end:]
            else:
                var_end = var_end_loc(namelist_string,param_key)
                namelist_string = namelist_string[:var_end] + scanwith[param_key] + namelist_string[var_end:]
                
        # placing in the remote save directory
        lines = namelist_string.split('\n')
        for line, i in zip(lines, np.arange(len(lines))):
            if 'diagdir' in line: lines[i] = f"    diagdir = '{self.remote_save_dir}'" 
        namelist_string = '\n'.join(lines)

        #Writing the final namelist stirng to file. This is the scan parameters file.
                # checking run dir exists and making Path for scan file
        print('Writing to', self.save_dir)
        if os.path.exists(self.save_dir):
            input_fpath = os.path.join(self.save_dir, file_name)
        else:
            raise FileNotFoundError(f'Couldnt find {self.save_dir}')

        with open(input_fpath, 'w') as file:
            file.write(namelist_string)  
        
        return namelist_string
    

    def read_output_file(self, out_path=os.path.join('/scratch/project_462000451/daniel/AUGUQ/scanfiles0002/scan.log')):
        growthrate = []
        frequency = []
        head = open(out_path, 'r').readline()
        # head.replace('\n','')
        head = head.split('|')
        last_two = head[-1].split('/')
        del head[-1]
        head = head + last_two
        head = [h.replace(' ','') for h in head]

        df = pd.read_csv(out_path, sep='|',skiprows=1, names=head)
        for i in range(len(df)):
            split = df[head[-1]][i].lstrip().rstrip().split(' ')      
            growthrate.append(split[0])
            frequency.append(split[-1])
        df['growthrate'] = growthrate
        df['frequency'] = frequency
        
        df = df.drop(columns=[head[0],head[-1]])
        return df
    
    def read_run_time(self, geneerr_path):
        # Open the file in read mode
        times = []
        with open(geneerr_path, 'r') as file:
            # Iterate through each line in the file
            for line_number, line in enumerate(file, start=1):
                # Check if the desired string is in the current line
                if 'Time for GENE simulation:' in line:
                    time = line.strip().split(' ')[-2]#re.search('(?<=Ti :)[^.\s]*',line)
                    times.append(time)
                    # print('TIME',time)
                    # print('LINE', type(line), line, type(line.strip()), line.strip())
                    # print(f"Line {line_number}: {line.strip()}")
        df = pd.DataFrame(times, columns=['run-time'])
        return df
    
    def read_scan_status(self, sbatch_out_path):
        scan_status = 'some error'
        with open(sbatch_out_path, 'r') as file:
            # Iterate through each line in the file
            for line_number, line in enumerate(file, start=1):
                if 'gene did not finish all runs, ready for continuation...' in line:
                    scan_status = 'ready for continuation'
                elif 'creating scan.log'in line:
                    scan_status = 'complete'
    
        return scan_status

    
if __name__ == '__main__':
    bounds = [[0.1, 300],[2,3.5],[4,6.8]]
    # params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1}
    generator = np.random.default_rng(seed=238476592)
    omn = generator.uniform(5,60,5)
    # 'box-kymin':generator.uniform(0.05,1,5)
    params = {'species-omn':omn,
          '_grp_species_1-omt':generator.uniform(10,70,5)}
    parser = GENE_scan_parser(save_dir= os.getcwd(),base_params_dir = os.path.join('/home/djdaniel/GENE_UQ/','parameters_base_uq'), remote_save_dir='/project/project_462000451/gene_out/gene_auto')
    # parser.alter_base(group_var="general_timelim",value=44000)
    # parser.write_input_file(params,file_name='parameters_scanwith')
    parser.read_run_time('scanlogs/5000s_7p/geneerr_batch-0_0.log')