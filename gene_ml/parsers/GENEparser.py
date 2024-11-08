import subprocess
import os
import f90nml.parser
import numpy as np
import f90nml
from typing import List
from copy import deepcopy
import pandas as pd
import re
import time
from ..tools import sec_to_time_format 

from collections import deque


import sys


# class GENEparser():
#     """An I/O parser for GENE

#      Attributes
#     ----------

#     Methods
#     -------
#     write_input_file
#         Writes the inputfile for a single set of parameters.

#     read_scanlog
#         Reads the output file to python format

#     """
#     def __init__(self, base_params_path=None, remote_save_dir=None):
#         """
#         Generates the base f90nml namelist from the GENE parameters file at base_params_path.

#         Parameters
#         ----------
#             base_params_path (string or path): The directory pointing to the base GENE parameters file.
#             The base GENE parameters file must contain all parameters necessary for GENE to run.
#             Any parameters to be sampled will be inserted into the base parameter file before each run.
#             Any value of a sampled parameter in the base file will be ignored. 
#         Returns
#         -------
#             Nothing 
#         """
#         if base_params_path!=None:
#             self.base_namelist = f90nml.read(base_params_path) #odict_keys(['parallelization', 'box', 'in_out', 'general', 'geometry', '_grp_species_0', '_grp_species_1', 'units'])
#         if remote_save_dir != None:
#             self.remote_save_dir = remote_save_dir

# class GENE_single_parser():

#     def write_input_file(self, params: dict, save_dir):
#         """
#         Write the GENE input file to the run directory specified. 
        
#         Parameters
#         ----------
#             params (dict): The keys store strings of the names of the parameters as specified in the enchanted surrogates *_config.yaml configuration file.
#             The values stores floats of the parameter values to be ran in GENE.

#             rprint('Writing to', save_dir)
#         if os.path.exists(save_dir):
#             input_fpath = os.path.join(save_dir, 'input.tglf')
#             subprocess.run(['touch', f'{input_fpath}'])
#         else:
#             raise FileNotFoundError(f'Couldnt find {save_dir}')un_dir (string or path): The file system directory where runs are to be stored

#         """
#         print('Writing to', save_dir)
#         if os.path.exists(save_dir):
#             self.input_fpath = os.path.join(save_dir, 'parameters')
#         else:
#             raise FileNotFoundError(f'Couldnt find {save_dir}')
        

#         params_keys = list(params.keys())
#         params_values = list(params.values())
#         patch = {}
#         patch['in_out'] = {'diagdir':self.remote_save_dir}
#         for key, value in zip(params_keys,params_values):
#             group_name, variable_name = key.split('-')
#             if list(patch.keys()).count(group_name) > 0:
#                 patch[group_name][variable_name] = value
#             else: patch[group_name] = {variable_name:value}


#         namelist = self.base_namelist
#         patch = f90nml.namelist.Namelist(patch)
#         namelist.patch(patch)
        
#         f90nml.write(namelist, self.input_fpath)

#     # what is returned here is returned to the runner for a single code run, which goes though the base executor to get to the future 
#     def read_scanlog(self, save_dir: str):
#         raise NotImplementedError
    
class GENE_scan_parser(): 
    def __init__(self, config):
        """
        Generates the base f90nml namelist from the GENE parameters file at base_params_path.

        Parameters
        ----------
            base_params_path (string or path): The directory pointing to the base GENE parameters file.
            The base GENE parameters file must contain all parameters necessary for GENE to run.
            Any parameters to be sampled will be inserted into the base parameter file before each run.
            Any value of a sampled parameter in the base file will be ignored. 
        Returns
        -------
            Nothing 
        """
        self.config = config
        # self.save_dir = config.save_dir
        self.base_params_path = config.base_params_path
        self.base_namelist = f90nml.read(self.base_params_path)
        self.base_sbatch_path = config.base_sbatch_path
        if str(type(self.base_params_path)) == "<class 'paramiko.sftp_file.SFTPFile'>":
            self.remote_base = True
        else:
            self.remote_base = False
    # def alter_base(self, group_var, value):
    #     #currently only works for variables that only appear in one group, not omn as it is in each species group
    #     #var example var="general_timelim", group_variable for fortran parameters file.
    #     group, var = group_var.split("_")
    #     patch = {group: {var: value}}

    #     self.base_namelist.patch(patch)

    #     print('Writing to', self.base_params_path)
    #     if os.path.exists(self.base_params_path) and not self.remote_base:
    #         f90nml.write(self.base_namelist, self.base_params_path, force=True)
    #     elif self.remote_base:
    #         self.base_params_path.write(str(self.base_namelist))
    #     else:
    #         raise FileNotFoundError(f'Couldnt find {self.base_params_path}')

    def get_parameter_value(self, parameters_path, group_var):
        print('gpv, paramater path', parameters_path)
        with self.open_file(parameters_path, 'r') as parameters_file:
            nml = f90nml.read(parameters_file)
        group, var = group_var
        return nml[group][var]
        
    # def alter_parameters_file_old(self, parameters_path, group_var, value):
    #     print('ALTER PARAMETERS FILE')
    #     # This will work to alter any parameter that does not have a comment in its line. eg scanned parameters
    #     new_parameters_file = []
    #     with self.open_file(parameters_path, 'r') as parameters_file:
    #         nml = f90nml.read(parameters_file)
    #         # print('old nml', nml)
    #         group, var = group_var
    #         patch = {group: {var: value}}
    #         nml.patch(patch)
            
    #     with self.open_file(parameters_path, 'r') as parameters_file:
    #         for line_old, line_new in zip(parameters_file, str(nml).splitlines()):
    #             if '!' in line_old: 
    #                 new_parameters_file.append(line_old)
    #             else: 
    #                 new_parameters_file.append(line_new)
    #     new_parameters_file.append('/')# I noticed it was missing, not sure why.
    #     new_parameters_file = '\n'.join(new_parameters_file)

    #     with self.open_file(parameters_path, 'w') as parameters_file:
    #         parameters_file.write(new_parameters_file)
    #     # print('new parameters file', new_parameters_file)
    #     return new_parameters_file        

    


    #commented as decided to keep gene units and use alter parameter instead. 
    # def set_simtimelim(self,simtimelim_sec, parameters_path, alter = True):
    #     #convert seconds to GENE units of cref/Lref
    #     # cref = sqrt(Tref / mref)
    #     #!!Caution assumes Tref is given in the base parameters file and will not work if set to -1 for GENE computation
        
    #     if simtimelim_sec == None:
    #         return None
    #     else:
    #         with self.open_file(parameters_path, 'r') as parameters_file:
    #             nml = f90nml.read(parameters_file)
    #             tref = nml['units']['tref']
    #             if tref < 0:
    #                 #raise ValueError('To set the simulation time limit based on Tref it must be included in the base parameters file. This is needed to compute the units.')
    #                 tref = 0.501585859244667 # taken from origional parameters file
    #                 print('!!WARNING!!, tref is being computed by gene but we do not know what it is, so using a default value of', tref)
    #             mref = nml['units']['mref']
    #             cref = np.sqrt(tref/mref)
    #             lref = nml['units']['lref']
                
    #             simtimelim_gene = simtimelim_sec / (lref / cref)
    #             print('SET_SIMTIMELIM, SIMTIMLIM_GENE:', simtimelim_gene)
    #             print('TREF', tref, 'MREF', mref, 'CREF',cref, 'LREF', lref)
    #             if alter: self.alter_parameters_file(parameters_path=parameters_path, group_var=["general","simtimelim"], value=simtimelim_gene)
    #     return simtimelim_gene

    def base_to_remote(self, remote_param_path, remote_sbatch_path):
        print('PLACING BASE PARAMETERS AND SBATCH TO REMOTE PROBLEM DIRECTORY')
        self.config.paramiko_sftp_client.put(self.base_params_path, remote_param_path)
        self.config.paramiko_sftp_client.put(self.base_sbatch_path, remote_sbatch_path)        

    def alter_parameters_file(self, parameters_path, group_var, value):
        group, var = group_var
        group_ord = re.findall('[0-9]', group)
        if len(group_ord) > 1:
            raise ValueError('Daniel Says, group names should only containc max one digit that specifies the ordinal of the species. If there are more than 9 species then the parser needs altered to accomodate.')
        elif len(group_ord) == 1:
            group_ord = int(group_ord[0])
            group_count = group_ord
        else: group_count = 0
        with self.open_file(parameters_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if ' '+var+' ' in line: # spaces ensure filenames are not picked
                    group_count -= 1
                if ' '+var+' ' in line and group_count < 0: # spaces ensure filenames are not picked     
                    print(i,var, value)
                    print(lines[i])
                    lines[i] = f'    {var} = {value}\n'
                    print(lines[i])
                    break # break at first occurence since there is only one
        new_parameters = ''.join(lines)
        with self.open_file(parameters_path, 'w') as file:
            file.write(new_parameters)

    def rename_important_scanfiles(self, scanfile_dir, prefix, important_files = ['scan.log', 'geneerr.log']):
        #renaming important files so they don't get overwritten
        print("RENAMING IMPORTANT FILES SO THEY DON'T GET OVERWRITTEN")
        important_files_new = [f'{prefix}_{i_f}' for i_f in important_files]
        important_paths = [os.path.join(scanfile_dir,i_f) for i_f in important_files]
        important_paths_new = [os.path.join(scanfile_dir, i_f_n) for i_f_n in important_files_new]
        for i_path, i_path_new in zip(important_paths, important_paths_new):
            self.config.paramiko_ssh_client.exec_command(f'mv {i_path} {i_path_new}')
        time.sleep(5) # To ensure the mv command has finished before moving on. Not doing this caused some timing issues with file reading things that had not been moved yet.


    def write_sbatch(self, sbatch_path, sbatch_continue_path, wallseconds):
        print('WRITE SBATCH')
        with self.open_file(sbatch_path, 'r') as sbatch_file:
            sbatch_lines = sbatch_file.readlines()
            wall_clock_limit = sec_to_time_format(wallseconds)
            wall_loc = 0
            for i in range(len(sbatch_lines)):
                if '#SBATCH -t' in sbatch_lines[i]: 
                    wall_loc = i
                    break
            sbatch_lines[wall_loc] = f"#SBATCH -t {wall_clock_limit}  ## wallclock limit, dd-hh:mm:ss\n"

            sbatch = "".join(sbatch_lines)

        with self.open_file(sbatch_path, 'w') as sbatch_file:
            sbatch_file.write(sbatch)

        # Make contiue scan script
        for i, line in enumerate(sbatch_lines):
            if "./scanscript" in line:
                sbatch_lines[i] = line.replace("./scanscript", "./scanscript --continue_scan")

        continue_str = "".join(sbatch_lines)
        with self.open_file(sbatch_continue_path, "w") as continue_file:
            continue_file.write(continue_str)
        return sbatch

    #puts in the paramaters with the GENE !scan functionality
    def write_input_file(self, params: dict, parameters_path, remote_save_dir):
        with self.open_file(parameters_path) as parameters_file:
            namelist = f90nml.read(parameters_file)
        # namelist = self.base_namelist
        namelist_string=str(namelist)
        
        #populate params: dict with all omn's required. Since each should be identical
        if 'species-omn' in params or ('species','omn') in params:
            print('species-omn was not handled by a converter, so we are handeling it now by making species 1 and 2 with the same omn')
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

            var_start = group_start+namelist_string[group_start:group_end].find(var_name+' ') #space ensures it is not apart of another name. Only works if there is a space after every variable
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
            if type(param_key) == type(('tuple','tuple')):
                group, var_name = param_key
            elif type(param_key) == type('string'):
                Warning("Daniel Warns, we are moving away from 'group_value' format and going towards ('group','value') format")
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
            if 'diagdir' in line: lines[i] = f"    diagdir = '{remote_save_dir}'" 
        namelist_string = '\n'.join(lines)

        #Writing the final namelist stirng to file. This is the scan parameters file.
                # checking run dir exists and making Path for scan file
        print('Writing to', parameters_path)
    
        with self.open_file(parameters_path, 'w') as file:
            file.write(namelist_string)  
        
        return namelist_string

    def listdir(self,dir):
        try:
            in_dir = self.config.paramiko_sftp_client.listdir(dir)
        except:
            in_dir=os.listdir(dir)
        return in_dir
    
    def read_species_names(self, parameters_path):
        names = []
        with self.open_file(parameters_path) as parameters_file:
            for line in parameters_file:
                if 'name' in line:
                    names.append(line.split('=')[-1].strip().strip("'")) 
        names = [s.replace('i', 'ion').replace('e', 'electron') for s in names]
        return names
    
    def read_fluxes(self, scanfiles_dir, nrg_prefix='', nspecies=2):
        print('READING FLUXES')
        species_names = self.read_species_names(parameters_path=os.path.join(scanfiles_dir,'parameters'))
        files = self.listdir(scanfiles_dir)
        nrg_files = np.sort(np.array([f for f in files if nrg_prefix+'nrg' in f]))
        
        # columns = []
        # for i in range(1,nspecies+1):
        #     columns += [f'particle_electrostatic_{i}', f'particle_electromagnetic_{i}', f'heat_electrostatic_{i}', f'heat_electromagnetic_{i}']
                
        # df = pd.DataFrame(columns=columns)
        df = pd.DataFrame()
        # for i in range(1,nspecies+1):
        #     df[f'flux_particle_electrostatic_{i}']


        # def read_last_lines(file_path, num_lines):
        #     with open(file_path, 'rb') as file:
        #         return deque(file, maxlen=num_lines)

        for index, nrg_f in enumerate(nrg_files):
            nrg_path = os.path.join(scanfiles_dir,nrg_f)
            with self.open_file(nrg_path, 'r') as nrg_file:
                lines = deque(nrg_file, maxlen=nspecies)
                #lines = nrg_file.readlines()[-nspecies:]
            species = []
            for species_name, l in zip(species_names, lines):
                values = re.findall("(-?\d+\.\d+E[+-]?\d+)", l)#np.array(l.split('  ')"
                fluxes = values[4:8]
                fluxes_df = pd.DataFrame({f'particle_electrostatic_{species_name}':float(fluxes[0]), f'particle_electromagnetic_{species_name}':float(fluxes[1]), f'heat_electrostatic_{species_name}':float(fluxes[2]), f'heat_electromagnetic_{species_name}':float(fluxes[3])},index = [index])
                species.append(fluxes_df)
            all_species = pd.concat(species, axis=1)
            df = pd.concat([df,all_species], axis=0)
            # The species are in the same order as in the gene_parameters file.
        
        return df
    
    def read_parameters_dict(self, parameters_path):
        # for some reason f90nml fails to parse with 'FCVERSION' line in the parameters file, so I comment it
        with self.open_file(parameters_path, 'r') as parameters_file:
            lines = parameters_file.readlines()
            for i, line in enumerate(lines):
                if 'FCVERSION' in line:
                    lines[i] = '!'+line

        with self.open_file(parameters_path, 'w') as parameters_file:
            parameters_file.writelines(lines)

        with self.open_file(parameters_path, 'r') as parameters_file:
            nml = f90nml.read(parameters_file)
            parameters_dict= nml.todict()
        return parameters_dict



    def read_scanlog(self, scanlog_path=os.path.join('/scratch/project_462000451/daniel/AUGUQ/scanfiles0002/scan.log')):
        growthrate = []
        frequency = []
        with self.open_file(scanlog_path, 'r') as scanlog_file:
            head = scanlog_file.readline()
        # head.replace('\n','')
        head = head.split('|')
        last_two = head[-1].split('/')
        del head[-1]
        head = head + last_two
        head = [h.replace(' ','') for h in head]

        with self.open_file(scanlog_path, 'r') as scanlog_file:
            df = pd.read_csv(scanlog_file, sep='|',skiprows=1, names=head)
    
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
        with self.open_file(geneerr_path, 'r') as file:
            # Iterate through each line in the file
            for line_number, line in enumerate(file, start=1):
                # Check if the desired string is in the current line
                if 'Time for GENE simulation:' in line:
                    time = line.strip().split(' ')[-2]#re.search('(?<=Ti :)[^.\s]*',line)
                    times.append(time)
                    # print('TIME',time)
                    # print('LINE', type(line), line, type(line.strip()), line.strip())
                    # print(f"Line {line_number}: {line.strip()}")
        df = pd.DataFrame(times, columns=['run_time'])
        return df
    
    def hit_wallclock_limit_test(self, sbatch_err_path):
        response = False
        with self.open_file(sbatch_err_path, 'r') as file:
            # Iterate through each line in the file
            for line_number, line in enumerate(file, start=1):
                if 'DUE TO TIME LIMIT ***' in line:
                    response = True
        return response
    
    def hit_simtimelim_test(self, geneerr_path, get_status=False, get_reasons=False, fast=False):
        print('HIT SIM LIMIT TEST ON FILE:', geneerr_path)
        response = None
        status = ''
        if fast:
            raise NotImplemented
            # need to perfect the regex for this.
            with self.open_file(geneerr_path, 'r') as geneerr:
                geneerr_str = str(geneerr.read())
                print('type',type(geneerr_str))
                match = np.array(re.findall("Simulation time limit of.*reached, exiting time loop|Linear growth rate is converged, exiting time loop", geneerr_str))
                print(match)
                mask = np.where(match == 'Linear growth rate is converged, exiting time loop', True, False)
                match[mask] = 'f'
                match[not mask] = 's'
                status = ''.join(match)
        else:        
            with self.open_file(geneerr_path, 'r') as geneerr:
                run_count = 0
                unknown_reason_count = 0
                reasons = []
                for i, line in enumerate(geneerr):
                    if '*** entering time loop ***' in line:
                        print('line', line)
                        print('hstlt, len stat and run_count',len(status), run_count)
                        run_count += 1
                        
                    if re.search("Simulation time limit of.*reached, exiting time loop", line) != None:
                        status += 's'
                        reasons.append('simtimelim')

                    elif re.search("Linear growth rate is converged, exiting time loop",line) != None:
                        status += 'f'
                        reasons.append('growthrate_converged')

                    elif re.search("Exit due to reaching the underflow limit",line) != None:
                        status += 'f'
                        reasons.append('underflow')

                    elif re.search('Exit due to overflow error', line) != None:
                        status += 'f'
                        reasons.append('overflow')

                if len(status) != run_count:
                    dif = run_count-len(status)
                    status += 's'*dif
                    # No reason given also included unknown reasons beyond the ones in if statements below
                    unknown_reason_count = dif
                        
                print(f'There have been {unknown_reason_count} unknown reasons for run termination found in {geneerr_path}')
        known_reasons = ['Simulation time limit of.*reached, exiting time loop', 'Linear growth rate is converged, exiting time loop', 'Exit due to reaching the underflow limit']
        print('known reasons for gene termination:', known_reasons )
        print('REASONS FOR GENE STOPPING DETECTED',np.unique(np.array(reasons)))

        if get_status:
            response = status
        elif get_reasons:
            response = reasons
        elif 's' in status:
            response = True
        else:
            response = False
        return response
    
    def check_status(self,status_path):
        with self.open_file(status_path, 'r') as status_file:
            status = status_file.read().decode('utf8')
            return status
    
    def set_status(self, status_path, status):
        with self.open_file(status_path, 'w') as status_file:
            status_file.write(status)
    # def read_scan_status(self, sbatch_err_path=None, gene_status_path=None):
    #     # this is mostly for checking to see if a scan needs to be immediatly continued because some of the runs are unfinished
    #     scan_status = 'complete or some error'
    #     if self.hit_wallclock_limit_test(sbatch_err_path):
    #         scan_status = 'needs continuation'
                
    #     return scan_status
    
    def open_file(self, file_path, mode='r'):
        if self.config.local_username in file_path:
            try: 
                file = open(file_path, mode)
            except:
                # print('using paramiko')
                file = self.config.paramiko_sftp_client.open(file_path, mode)
        else:
            try:
                # print('using paramiko 2')
                file = self.config.paramiko_sftp_client.open(file_path, mode)
            except:
                file = open(file_path, mode)
        return file

    
if __name__ == '__main__':
    bounds = [[0.1, 300],[2,3.5],[4,6.8]]
    # params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1}
    generator = np.random.default_rng(seed=238476592)
    omn = generator.uniform(5,60,5)
    # 'box-kymin':generator.uniform(0.05,1,5)
    params = {'species-omn':omn,
          '_grp_species_1-omt':generator.uniform(10,70,5)}
    parser = GENE_scan_parser(save_dir= os.getcwd(),base_params_path = os.path.join('/home/djdaniel/GENE_UQ/','parameters_base_uq'))
    # parser.alter_base(group_var="general_timelim",value=44000)
    # parser.write_input_file(params,file_name='parameters_scanwith',  remote_save_dir='/project/project_462000451/gene_out/gene_auto')
    # parser.read_run_time('scanlogs/5000s_7p/geneerr_batch-0_0.log')
    # parser.set_simtimelim(10e-2)
    parser.open_file('.gitignore')