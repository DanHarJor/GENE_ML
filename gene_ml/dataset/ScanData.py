#test edit2
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import warnings
import re
try:
    from .base import DataSet
except:
    try:
        from base import DataSet
    except:
        raise ImportError 

sys.path.append(os.path.join('GENE_ML','IFS_scripts'))
from GENE_ML.IFS_scripts.geomWrapper import calc_kperp_omd, init_read_geometry_file
from GENE_ML.IFS_scripts.parIOWrapper import init_read_parameters_file
    

class ScanData(DataSet):
    def __init__(self, name, parser, config, sampler=None, host=None, remote_save_dir=None, scan_name='', test_percentage=50, random_state=47, parameters_map=None, scan_files_target = 'all'):
        '''
        To retrieve data from the server remote path and host should be defined
        When the string ssh <host> is entered in to he command line a ssh terminal should be started.
        The remote path needs needs to point to either a diectory that contains the scanfile* folders from a GENE scan or a specific scan.log file.
        '''
        print('Initialising dataset')
        self.config = config
        self.scan_files_target = scan_files_target
        self.name = name
        self.parser = parser
        self.host=host
        self.remote_save_dir = remote_save_dir
        self.scan_name = scan_name
        self.sampler = sampler
        self.test_percentage=test_percentage

        self.random_state=random_state
        self.parameters_map = parameters_map

        self.ssh_path = f"{self.host}:{self.remote_save_dir}"
        print('SSH PATH', self.ssh_path)
        self.scan_log_path = os.path.join(os.getcwd(), 'scanlogs', self.name)
    
        # self.scan_log_dir = os.path.join(os.getcwd(), 'scanlogs')

        if not os.path.exists(os.path.join(os.getcwd(), 'scanlogs')):
            os.mkdir(os.path.join(os.getcwd(), 'scanlogs'))

        print('SCAN LOG PATH', self.scan_log_path)

        if not os.path.exists(self.scan_log_path): 
            print('MAKING SCANLOG DIR') 
            os.mkdir(self.scan_log_path)
            print('RETRIEVING REMOTE SCANLOG FILES')
            self.retrieve_remote_logs(scanfiles_target=self.scan_files_target)

        self.df_inc_nan = None
        if os.path.isfile(self.scan_log_path):
            print('\nLOADING FROM SCANLOG FILE')
            self.df_inc_nan = self.load_from_file(self.scan_log_path)
            self.df, n_samp, n_requested, n_samp_nonan = self.remove_nans(self.df_inc_nan)
        else:
            print(f'\nLOADING FROM SCANLOG DIR: {self.scan_log_path}')
            self.df, n_samp, n_requested, n_samp_nonan = self.load_from_dir(self.scan_log_path)

        print(f'\n{n_samp} SAMPLES RAN OUT OF {n_requested} BEFORE MAX WALLTIME:')
        print("NUMBER OF SAMPLES AFTER REMOVING NaN's:", n_samp_nonan)
        nan_percentage = (n_samp-n_samp_nonan)*100/n_samp
        print('NaN PERCENTAGE = ', nan_percentage)
        
        #for quasineutrality omn1 is the same as omn2 so we can remove one
        if any(np.array(self.df.columns.values.tolist()) == 'omn2'):
            self.df = self.df.drop(columns=['omn2'])
        if type(self.sampler) != type(None):        
            self.set_from_df()
            self.match_sampler(self.sampler)
        print('SETTING VARIABLES')
        self.set_from_df()
        
    def set_from_df(self):
        self.head = list(self.df.columns)
        self.x = self.df.drop(columns=['growthrate','frequency','run_time']).to_numpy(dtype=float)#self.df[self.head[0:-2]].to_numpy(dtype=float)
        self.growthrates = self.df['growthrate'].to_numpy(dtype=float)
        self.frequencies = self.df['frequency'].to_numpy(dtype=float)
        self.run_time = self.df['run_time'].to_numpy(dtype=float)
        if self.test_percentage == 0:
            print('TEST PERCENTAGE IS 0, NO SPLIT')
            self.x_train = self.x
            self.x_test = None
            self.growthrate_train = self.growthrates
            self.growthrate_test = None
            self.frequencies_train = self.frequencies
            self.frequencies_test = None
        else:
            self.split()

    def split(self):    
        print(f'\nRANDOMLY SPLITTING DATA INTO TEST AND TRAINING SETS: {self.test_percentage}% test, {100-self.test_percentage} training.')
        self.x_train, self.x_test, self.growthrate_train, self.growthrate_test, self.frequencies_train, self.frequencies_test = train_test_split(self.x, self.growthrates, self.frequencies, test_size=self.test_percentage/100, random_state=self.random_state)


    def load_from_file(self, scan_path, geneerr_path):
        print(f'\nLOADING SCANLOG AND TIME INTO PANDAS DATAFRAME {scan_path} : {geneerr_path}')
        scan_df = self.parser.read_scanlog(scan_path)
        if type(geneerr_path) == type(None):
            time_df = pd.DataFrame(np.repeat(np.nan, len(scan_df)), columns=['run_time'])
        else:
            time_df = self.parser.read_run_time(geneerr_path)
        df = pd.concat([time_df, scan_df], axis=1)
        return df
    
    # def retrieve_out_file(self, fname):
    #     save_path = os.path.join(os.getcwd(), 'out_files', self.name, fname)
    #     if not os.path.exists(save_path):
    #         os.system(f'mkdir -p {save_path}')
    #     print(f'\nRETRIEVING {fname} VIA scp FROM REMOTE')
    #     i = 0
    #     results = []
    #     while True:
    #         j=0
    #         while True:
    #             results.append(os.system(f"scp '{self.ssh_path}/*batch*{i}/scanfiles*{j}/{fname}' {os.path.join(save_path,f'{fname}_{i}_{j}.log')}"))
    #             print(results)
    #             j+=1
    #             if results[-3:] == [256,256,256]:
    #                 break
    #         # result = os.system(f"scp 'lumi:/scratch/project_462000451/gene_out/gene_auto/testing_batchscans/scanfiles*{i}/scan.log' $PWD/scanlogs/testing_batchscans/scan{i}.log")
    #         i+=1
    #         if results[-6:] == [256,256,256,256,256,256]:
    #             break 
    #     print('THE ABOVE ERROR IS EXPECTED')
    
    #     print(f'ALL {fname} RETRIEVED AND SAVED TO:{save_path}')
        
        

    def retrieve_remote_logs(self, scanfiles_target = 'all'):
        #scanfiles target can be 'all' or 'latest'
        print('\nRETRIEVING SCANLOG/S VIA scp FROM REMOTE')
        print(f'SCANLOG PATH: {self.ssh_path}')
        # subprocess.run(['scp','-r',ssh_path, self.scan_log_path])
        # print('\n\nDEBUG\n\n',self.scan_log_path, type(self.scan_log_path), str(self.scan_log_path))
        if 'scan.log' in self.ssh_path: #Then we are only taking one file
            print('RETRIVING SPECIFIED SINGLE LOGFILE')
            print('RETRIVING REMOTE FILE')
            os.system(f'scp -r {self.ssh_path} {self.scan_log_path}')
        elif scanfiles_target=='all': # we should have a directory and need to take all scan logs
            print('GETTING ALL SCANLOGS')
            print('RETRIVING FROM REMOTE DIR')
            i = 0
            results = []
            while True:
                j=0
                while True:
                    results.append(os.system(f"scp '{self.ssh_path}/*batch*{i}/scanfiles*{j}/scan.log' {os.path.join(self.scan_log_path,f'scan_batch-{i}_scanfiles-{j}.log')}"))
                    os.system(f"scp '{self.ssh_path}/*batch*{i}/scanfiles*{j}/geneerr.log' {os.path.join(self.scan_log_path,f'geneerr_batch-{i}_scanfiles-{j}.log')}")
                    # results.append(os.system(f"scp '{self.ssh_path}/scanfiles*{j}/scan.log' {os.path.join(self.scan_log_path,f'scan_batch-{i}_{j}.log')}"))
                    # os.system(f"scp '{self.ssh_path}/scanfiles*{j}/geneerr.log' {os.path.join(self.scan_log_path,f'geneerr_batch-{i}_{j}.log')}")

                    print(results)
                    j+=1
                    if results[-3:] == [256,256,256]:
                        break
                # result = os.system(f"scp 'lumi:/scratch/project_462000451/gene_out/gene_auto/testing_batchscans/scanfiles*{i}/scan.log' $PWD/scanlogs/testing_batchscans/scan{i}.log")
                i+=1
                if results[-6:] == [256,256,256,256,256,256]:
                    break 
            print('THE ABOVE ERROR IS EXPECTED')
        elif scanfiles_target =='latest':
            print('GETTING LATEST SCANLOG')
            #Using new paramiko library
            run_ids = np.sort(np.array(self.listdir(self.remote_save_dir)))
            print('RUN_IDs',run_ids)
            for i, r_id in enumerate(run_ids):
                scanfiles_dir = self.config.paramiko_sftp_client.listdir(os.path.join(self.remote_save_dir,r_id))
                print('SCANFILES_DIR', scanfiles_dir)
                scanfiles_number = [re.findall('[0-9]{4}',sc_dir) for sc_dir in scanfiles_dir]
                print('SCANFILES_NUMBER', scanfiles_number)
                latest_scanfile = scanfiles_dir[np.argmax(np.array(scanfiles_number).astype('int'))]
                print('LATEST_SCANFILE', latest_scanfile)
                latest_scanlog_path = os.path.join(self.remote_save_dir,r_id,latest_scanfile,'scan.log')
                latest_generr_path = os.path.join(self.remote_save_dir,r_id,latest_scanfile,'geneerr.log')
                self.config.paramiko_sftp_client.get(latest_scanlog_path, os.path.join(self.scan_log_path,f'scan_batch-{i}_scanfiles-0.log'))
                self.config.paramiko_sftp_client.get(latest_generr_path, os.path.join(self.scan_log_path,f'geneerr_batch-{i}_scanfiles-0.log'))



        print(f'SCANLOG/S RETRIEVED AND SAVED TO:{self.scan_log_path}')
        

    def load_from_dir(self, scan_path):
        '''
        This function assumes the directory only contains log files
        '''
        if not os.path.isdir(scan_path): raise NotADirectoryError
        
        dfs, n_samp_all, n_requested_all, n_samp_nonan_all = [], [], [], []
        dfs_inc_nans = []

        log_paths = np.sort(np.array(os.listdir(scan_path)))
        print('LOG PATHS',log_paths)
        geneerr_paths = [os.path.join(scan_path,p) for p in log_paths if 'geneerr' in p]
        scanlog_paths = [os.path.join(scan_path,p) for p in log_paths if 'scan_' in p]
        if len(geneerr_paths) != len(scanlog_paths):
            geneerr_paths = [None]*len(scanlog_paths)

        for scanlog_path, geneerr_path in zip(scanlog_paths, geneerr_paths):
            df = self.load_from_file(scanlog_path, geneerr_path)
            dfs_inc_nans.append(df)
            df, n_samp, n_requested, n_samp_nonan = self.remove_nans(df)
            dfs.append(df); n_samp_all.append(n_samp); n_requested_all.append(n_requested); n_samp_nonan_all.append(n_samp_nonan)
        try:
            self.df_inc_nan = pd.concat(dfs_inc_nans)
        except: 
            raise ValueError('Daniel Jordan Says: There is nothing to concatenate into the dataframe. This could be because the scanlog folder is empty or has the incorrect syntax.')
        return pd.concat(dfs), np.sum(n_samp_all), np.sum(n_requested_all), np.sum(n_samp_nonan_all)
        
    def remove_nans(self, df):
        ## caution, can only work for df created from single file.
        #removing NAN's
        nan_mask = ~np.isnan(df['growthrate'].to_numpy(dtype=float))
        if len(np.argwhere(nan_mask))>0: 
            n_before_tlimit = int(np.argwhere(nan_mask)[-1])+1 
        else: 
            n_before_tlimit = 0
        n_requested = len(df)
        df = df[0:n_before_tlimit]
        nan_mask = nan_mask[0:n_before_tlimit]
        n_samp = len(df)
        df = df.loc[nan_mask]
        n_samp_nonan = len(df)
        return df, n_samp, n_requested, n_samp_nonan
    
    def match_sampler(self, sampler):
        print("\nCHECKING THAT THE SSG SAMPLER AND DATASET HAVE MATCHING ORDER OF SAMPLES...")
        # Check that the data and sampler have the same order________
        sample_order_bool = []
        print('x', self.x[0], 's', sampler.samples_array[0])
        print('l',len(self.x))
        for i in range(len(self.x)):
            sampler.samples_array[i]
            self.growthrate_train[i]
            #print(self.x[i], sampler.samples_array[i], f'gr {self.growthrate_train[i]}')
            #The GENE scanlogs don't have the parameters in the same order as the sampler
            #So we just check if they have the same numbers regardless of the order with sort
            # Also the scanlogs have both omn while the sampler just has one since they are the same (to conserve quasineutrality)
            # This is why the unique is there, to remove the duplicated omn. 
            #print('EQUAL??', np.sort(np.unique(np.round(self.x[i],2))), np.sort(np.unique(np.round(sampler.samples_array[i],2))))
            order_bool = np.sort(np.unique(np.round(self.x[i],3))) == np.sort(np.unique(np.round(sampler.samples_array[i],3)))
            sample_order_bool.append(order_bool)
        sample_order_bool = np.concatenate(sample_order_bool)
        all_corect_order = all(sample_order_bool)
        print("RESULT: ", all_corect_order, "\n")
        #_____________________________________________________________
            
        if not all_corect_order:
            print('The ssg_sampler.samples_array and the ssg_dataset.x have samples that are not in the same order. Thus ssg_poly cannot work.')
            raise KeyError
        else:
            print("\n MATCHING THE DATASET SAMPLES TO THE SAMPLERS SAMPLES. THIS IS BECAUSE THEY CAN HAVE DIFFERENT COLUMN CONVENTIONS.")
            # Now the order is correct we can safely make them the same. 
            new_df = sampler.df.copy()
            new_df['growthrate'] = self.df['growthrate'].to_numpy()
            new_df['frequency'] = self.df['frequency'].to_numpy()
            new_df.insert(0, 'run_time', self.df['run_time'].to_numpy())
            self.df = new_df
            self.set_from_df()
            print("COMPLETE \n")


    def match_parameters_order(self, parameters):
        ordered_head = [] 
        for param in parameters:
            ordered_head.append(self.parameters_map[param])
        self.df = self.df.loc[:,['run_time',*ordered_head,'growthrate','frequency']]
        
    def remove_parameter(self, parameter_name):
        self.df = self.df.drop(columns=[parameter_name])
        self.set_from_df()

    def train_time_model(self, model):
        self.time_model = model
        self.time_model.train(self.x, self.run_time)

    def concat(self, datasets):
        dfs = [self.df]
        for ds in datasets:
            dfs.append(ds.df)
        joint_df = pd.concat(dfs)
        self.df = joint_df
        self.set_from_df()
        return self
    
    def save(self, path):
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)



class ScanData2(DataSet):
    # paramiko update
    # update to include info about termination reason
    def __init__(self, config, name=None, parser=None, sampler=None, save_dir=None, scan_name='', split_ratio=[0.4,0.1,0.5], random_state=47, parameters_map=None, categorise=False):
        '''
        To retrieve data from the server remote path and host should be defined
        When the string ssh <host> is entered in to he command line a ssh terminal should be started.
        The remote path needs needs to point to either a diectory that contains the scanfile* folders from a GENE scan or a specific scan.log file.
        '''
        print('Initialising dataset')
        self.config = config
        self.name = name
        self.parser = parser
        self.save_dir = save_dir
        self.scan_name = scan_name
        self.sampler = sampler
        self.split_ratio = split_ratio

        self.random_state=random_state
        self.parameters_map = parameters_map
        self.categorise = categorise
        
        if save_dir != None:
            self.scanlog_df, self.rest_df = self.load_from_save_dir()
            self.rest_df_ncol=len(self.rest_df.columns)
            #for quasineutrality omn1 is the same as omn2 so we can remove one
            if any(np.array(self.scanlog_df.columns.values.tolist()) == 'omn2'):
                self.scanlog_df = self.scanlog_df.drop(columns=['omn2'])
            if any(np.array(self.rest_df.columns.values.tolist()) == 'omn2'):
                self.rest_df = self.rest_df.drop(columns=['omn2'])
                self.rest_df_ncol = len(self.rest_df.columns)

            print('SETTING VARIABLES')
            self.set_from_df()

        # self.df = pd.concat([self.scanlog_df, self.rest_df],axis=1)
        print('End of SCAN DATA init')

        # if type(self.sampler) != type(None):            
        #     self.match_sampler(self.sampler)
        
    def set_from_df(self, df=None, rest_df_ncol = 1):
        # is statement determins is we are setting to in inputed df or from the ones taken from files in the __init__
        if type(df) != type(None):
            if rest_df_ncol != None:
                self.rest_df_ncol = rest_df_ncol 
            self.rest_df = df.iloc[:, -self.rest_df_ncol:]
            self.scanlog_df = df.iloc[:, :-self.rest_df_ncol]
        
        self.df = pd.concat([self.scanlog_df, self.rest_df],axis=1)
        nan_mask = ~np.isnan(self.scanlog_df['growthrate'].to_numpy(dtype=float))
        self.scanlog_df_no_nan = self.scanlog_df.loc[nan_mask]
        self.rest_df_no_nan = self.rest_df.loc[nan_mask]
        self.df_no_nan = self.df[nan_mask]
        self.head = list(self.scanlog_df_no_nan.columns)
        self.x = self.scanlog_df_no_nan.drop(columns=['growthrate','frequency']).to_numpy(dtype=float)#self.df[self.head[0:-2]].to_numpy(dtype=float)
        self.growthrates = self.scanlog_df_no_nan['growthrate'].to_numpy(dtype=float)
        self.frequencies = self.scanlog_df_no_nan['frequency'].to_numpy(dtype=float)
        # self.run_time = self.rest_df_no_nan['run_time'].to_numpy(dtype=float)

        # Define the categories and their corresponding probabilities
        categories = ['train', 'val', 'test']
        # Generate a random sample of 10 categories based on the specified distribution
        random_categories = np.random.choice(categories, size=len(self.scanlog_df_no_nan), p=self.split_ratio)
        self.rest_df_no_nan['data_categorie'] = random_categories
        self.x_train = self.x[random_categories=='train']
        self.growthrate_train = self.growthrates[random_categories=='train']
        self.frequencies_train = self.frequencies[random_categories=='train']
        self.run_time_train = self.frequencies[random_categories=='train']

        self.x_val = self.x[random_categories=='val']
        self.growthrate_val = self.growthrates[random_categories=='val']
        self.frequencies_val = self.frequencies[random_categories=='val']
        self.run_time_val = self.frequencies[random_categories=='val']

        self.x_test = self.x[random_categories=='test']
        self.growthrate_test = self.growthrates[random_categories=='test']
        self.frequencies_test = self.frequencies[random_categories=='test']
        self.run_time_test = self.frequencies[random_categories=='test']

    def load_diffusivities(self, scanfiles_path):
        file_names = np.array(self.listdir(scanfiles_path))
        parameters_file_names = np.array([f for f in file_names if 'parameters_' in f and not 'eff' in f])
        parameters_file_names = np.sort(parameters_file_names)
        parameters_paths = [os.path.join(scanfiles_path, parameters_file_name) for parameters_file_name in parameters_file_names]
        fluxes_df = self.parser.read_fluxes(scanfiles_path)
        df = pd.DataFrame()
        for i, parameters_path in enumerate(parameters_paths):
            print(parameters_path)
            parameters_dict = self.parser.read_parameters_dict(parameters_path)
            nref = parameters_dict['units']['nref']
            Tref = parameters_dict['units']['tref']
            Lref = parameters_dict['units']['lref']
            keys = list(parameters_dict.keys())
            species = [s for s in keys if 'species' in s]
            species_df = []
            species_names = self.parser.read_species_names(parameters_path)
            for name, spec in zip(species_names,species):
                particle_flux = fluxes_df[f'particle_electrostatic_{name}'].iloc[i] + fluxes_df[f'particle_electromagnetic_{name}'].iloc[i]
                omn = parameters_dict[spec]['omn']
                n = nref * parameters_dict[spec]['dens']
                grad_n = -(n/Lref) * omn
                particle_diff = - particle_flux / grad_n
                
                heat_flux = fluxes_df[f'heat_electrostatic_{name}'].iloc[i] + fluxes_df[f'heat_electromagnetic_{name}'].iloc[i]
                omt = parameters_dict[spec]['omt'] 
                T = parameters_dict[spec]['temp'] * Tref
                grad_T = omt * -(T/Lref)
                
                heat_diff = -(heat_flux - (3/2)*T*particle_flux)/(n * grad_T)
                
                diff_df = pd.DataFrame({f'particle_diff_{name}':particle_diff, f'heat_diff_{name}':heat_diff, f'particle_flux_{name}':particle_flux, f'heat_flux_{name}':heat_flux}, index=[i])
                species_df.append(diff_df)
            all_species = pd.concat(species_df, axis=1)
            df = pd.concat([df,all_species], axis=0)
        return df
    
    def fingerprints_categorisation(self, scanfiles_path):
        # this method includes the fingerprints ratios and also includes the electron EM flux vs ES flux
        diff_df = self.load_diffusivities(scanfiles_path)
        print('DEBUG',diff_df.columns)
        heat_diff_i = diff_df['heat_diff_ion'].to_numpy()
        print('DEBUG',diff_df.columns)
        
        heat_diff_e = diff_df['heat_diff_electron'].to_numpy()
        ratio_iheat_eheat = heat_diff_i / heat_diff_e

        particle_diff_e = diff_df['particle_diff_electron'].to_numpy()
        ratio_eparticle_eheat = particle_diff_e / heat_diff_e

        ratio_eparticle_heat = particle_diff_e / (heat_diff_e + heat_diff_i)
        instability_rows = []
        
        # for i in range(len(diff_df)):
        #     if 0.2 < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 10 and 0.2 < ratio_eparticle_eheat[i] and ratio_eparticle_eheat[i] < 10:
        #         inst.append('MHD-like')
        #     if ratio_iheat_eheat[i] > 10:
        #         print(f"row {i} has a really high iheat_eheat flux ratio, over {10}, investigate")
        #     if ratio_eparticle_eheat[i] > 10:
        #         print(f"row {i} has a really high ratio_eparticle_eheat, over {10}, investigate")
        #     if 0 < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 0.25 and 0 < ratio_eparticle_eheat[i] and ratio_eparticle_eheat[i] < 0.66:
        #         inst.append('MTM')
        #     if 0.1-0.1*tolerance < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 0.1+0.1*tolerance and 0.05-0.05*tolerance < ratio_eparticle_eheat[i] and ratio_eparticle_eheat[i] < 0.05+0.05*tolerance:
        #         inst.append('ETG')
        #     if 0.25 < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 1 and -0.1-(1/3) < ratio_eparticle_heat[i] and ratio_eparticle_heat[i] < -0.1+(1/3):
        #         inst.append('ITG/TEM')
        #     else:
        #         inst.append('None')
        MHD_tolerance = 0.5
        MTM_tolerance = 0.1
        ETG_tolerance = 0.1
        ITG_TEM_tolerance = 0.1
        for i in range(len(diff_df)):
            inst = []
            if 1-MHD_tolerance < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 1+MHD_tolerance and (2/3)-MHD_tolerance < ratio_eparticle_eheat[i] and ratio_eparticle_eheat[i] < (2/3)+MHD_tolerance:
                inst.append('MHD-like')
            if 0.1-MTM_tolerance < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 0.1+MTM_tolerance and 0.1-MTM_tolerance < np.abs(ratio_eparticle_eheat[i]) and np.abs(ratio_eparticle_eheat[i]) < 0.1+MTM_tolerance:
                inst.append('MTM')
            if 0.1-ETG_tolerance < np.abs(ratio_iheat_eheat[i]) and ratio_iheat_eheat[i] < 0.1+ETG_tolerance and 0.05-ETG_tolerance < np.abs(ratio_eparticle_eheat[i]) and np.abs(ratio_eparticle_eheat[i]) < 0.05+ETG_tolerance:
                inst.append('ETG')
            if 0.25-ITG_TEM_tolerance < ratio_iheat_eheat[i] and ratio_iheat_eheat[i] < 1+ITG_TEM_tolerance and -0.1-(1/3)-ITG_TEM_tolerance < ratio_eparticle_heat[i] and ratio_eparticle_heat[i] < -0.1+(1/3)+ITG_TEM_tolerance:
                inst.append('ITG/TEM')
            if len(inst) == 0:
                inst.append('None')
            instability_rows.append(inst)

        df = pd.DataFrame()
        df['fingerprint'] = instability_rows
        df['ratio_iheat_eheat'] = ratio_iheat_eheat
        df['ratio_eparticle_eheat'] = ratio_eparticle_eheat
        return df
        # parameters_dict = self.parser.read_parameters_dict()
    
    def em_categorisation(self, scanfiles_path):
        # MTM has a lot of electromagnetic flux
        fluxes_df = self.parser.read_fluxes(scanfiles_path)
        ratio = fluxes_df['heat_electromagnetic_electron'] / fluxes_df['heat_electrostatic_electron']
        mask_MTM = ratio > 0.5
        categorisation = np.where(mask_MTM, 'yes_MTM', 'no_MTM')
        em_cat_df = pd.DataFrame(categorisation, columns=['em_categorisation'])
        return em_cat_df
    
    def final_categorisation(self, scanfiles_path):
        em_df = self.em_categorisation(scanfiles_path)
        fingerprints_df = self.fingerprints_categorisation(scanfiles_path)



        mask_mtm_fingerprint = np.array(['MTM' in row for row in fingerprints_df['fingerprint']])
        mask_em = np.array(['yes_MTM' in row for row in em_df['em_categorisation']])
        # mask_em = np.char.find(em_df['em_categorisation'], 'MTM') != -1 # might be faster, not tested
        mask_mtm = mask_mtm_fingerprint * mask_em
        
        mask_etg_fingerprint = np.array(['ETG' in row for row in fingerprints_df['fingerprint']])
        mask_em_flip = ~mask_em
        mask_etg = mask_etg_fingerprint * mask_em_flip
        
        categorisation = np.repeat('Nothing',len(fingerprints_df))

        overlap = mask_etg & mask_mtm
        if any(overlap):
            print('Daniel Says: There is an issue with the categorisation because some runs are being identified as multiple modes but only one it being returned')

        categorisation[mask_mtm] = 'MTM'
        categorisation[mask_etg] = 'ETG'

        return pd.DataFrame({'categorisation':categorisation}) 
    
    
    def kperp(self, scanfiles_path):
        # only set up to work for local files
        with self.parser.open_file(os.path.join(scanfiles_path, 'scan.log'), 'r') as file:
            line_count = sum(1 for line in file)
        # number_of_runs = line_count-1#same as number of lines in scan.log
        os.chdir(scanfiles_path)
        suffix_s = ['_'+str(number).zfill(4) for number in range(1,line_count)]

        avg_kperp_squared_s = []
        for suffix in suffix_s:
            pars = init_read_parameters_file(suffix)
            geom_type, geom_pars, geom_coeff = init_read_geometry_file(suffix,pars)

            kperp, omd_curv, omd_gradB = calc_kperp_omd(geom_type,geom_coeff,pars,False,False)
            
            avg_kperp_squared = np.mean(np.array(kperp)**2)
            avg_kperp_squared_s.append(avg_kperp_squared)
        
        kperp_df = pd.DataFrame({'avg_kperp_squared':avg_kperp_squared_s})
        return kperp_df

        
    

    def load_from_file(self, scanlog_path, geneerr_path, scanfiles_dir, nrg_prefix=''):    
        scanlog_df = self.parser.read_scanlog(scanlog_path)
        # Currently this assumes the generr file is in the same order as the scanlog file. It is known this is not true and an identifier needs to be placed to match them up.
        if self.categorise: 
            time_df = self.parser.read_run_time(geneerr_path)
            reasons = self.parser.hit_simtimelim_test(geneerr_path, get_reasons=True)
            reasons_df = pd.DataFrame(reasons, columns=['termination_reason'])
            fingerprints_df = self.fingerprints_categorisation(scanfiles_dir)
            em_cat_df = self.em_categorisation(scanfiles_dir)
            diff_df = self.load_diffusivities(scanfiles_dir)
            fluxes_df = self.parser.read_fluxes(scanfiles_dir)
            categorise_df = self.final_categorisation(scanfiles_dir)
            kperp_df = self.kperp(scanfiles_dir)            
            mixing_length_df = pd.DataFrame()
            mixing_length_df['mixing_length'] = scanlog_df['growthrate'].to_numpy(dtype=float)/kperp_df['avg_kperp_squared']

            rest_df = pd.concat([time_df, reasons_df, fingerprints_df, em_cat_df, diff_df, fluxes_df, categorise_df, kperp_df, mixing_length_df], axis=1)
            # print('DEBUG, time_df, reasons_df, rest, scanlog',len(time_df),len(reasons_df), len(rest_df), len(scanlog_df))
            # print('DEBUG',f'number of runs detected in the scan.log ({len(scanlog_df)}) --- geneerr.log ({len(rest_df)},{len(time_df)}, {len(reasons_df)}, {len(reasons)}).\n', scanlog_path, geneerr_path)

        else: rest_df = pd.DataFrame({'REST':np.repeat('rest', len(scanlog_df))})


        if len(scanlog_df) != len(rest_df):
            print(scanlog_df)
            print(rest_df)
            rest_df = pd.DataFrame({'ERROR':np.repeat(None, len(scanlog_df))})
            warnings.warn(f'Daniel Says: For some reason the number of runs detected in the scan.log ({len(scanlog_df)}) does not match geneerr.log ({len(rest_df)},{len(time_df)}, {len(reasons_df)}, {len(reasons)}).\n, {scanlog_path, geneerr_path}')
        
        return scanlog_df, rest_df
    
    def load_from_save_dir(self):
        batches = self.listdir(self.save_dir)
        batches = np.sort(np.array(batches))
        batch_dirs = [os.path.join(self.save_dir, batch) for batch in batches]
        latest_scanfile_dirs = []
        for batch_dir in batch_dirs:
            scanfiles_dir = np.sort(np.array(self.listdir(batch_dir)))
            scanfiles_number = [re.findall('[0-9]{4}',sc_dir) for sc_dir in scanfiles_dir]
            latest_scanfile = scanfiles_dir[np.argmax(np.array(scanfiles_number).astype('int'))]
            latest_scanfile_dirs.append(os.path.join(batch_dir,latest_scanfile))

        scanlog_dfs = []
        rest_dfs = []
        for scanfiles_dir in latest_scanfile_dirs:
            scanlog_path = os.path.join(scanfiles_dir,f'{self.scan_name}scan.log')
            generr_path = os.path.join(scanfiles_dir,f'{self.scan_name}geneerr.log')
            print('DEBUG2',scanfiles_dir)
            scanlog_df, rest_df = self.load_from_file(scanlog_path, generr_path, scanfiles_dir=scanfiles_dir)
            scanlog_dfs.append(scanlog_df)
            rest_dfs.append(rest_df)
        
        scanlog_df = pd.concat(scanlog_dfs, axis=0)
        scanlog_df.index = np.arange(len(scanlog_df))
        rest_df = pd.concat(rest_dfs, axis=0)
        rest_df.index = np.arange(len(rest_df))
        return scanlog_df, rest_df
    
    def listdir(self,dir):
        try:
            in_dir = self.config.paramiko_sftp_client.listdir(dir)
        except:
            in_dir=os.listdir(dir)
        return in_dir

    def remove_nans(self, df):
        ## caution, can only work for df created from single file.
        #removing NAN's
        nan_mask = ~np.isnan(df['growthrate'].to_numpy(dtype=float))
        if len(np.argwhere(nan_mask))>0: 
            n_before_tlimit = int(np.argwhere(nan_mask)[-1])+1 
        else: 
            n_before_tlimit = 0
        n_requested = len(df)
        df = df[0:n_before_tlimit]
        nan_mask = nan_mask[0:n_before_tlimit]
        n_samp = len(df)
        df = df.loc[nan_mask]
        n_samp_nonan = len(df)
        return df, n_samp, n_requested, n_samp_nonan
    
    def match_sampler(self, sampler):
        raise NotImplemented
        # print("\nCHECKING THAT THE SSG SAMPLER AND DATASET HAVE MATCHING ORDER OF SAMPLES...")
        # # Check that the data and sampler have the same order________
        # sample_order_bool = []
        # print('x', self.x[0], 's', sampler.samples_array[0])
        # print('l',len(self.x))
        # for i in range(len(self.x)):
        #     sampler.samples_array[i]
        #     self.growthrate_train[i]
        #     #print(self.x[i], sampler.samples_array[i], f'gr {self.growthrate_train[i]}')
        #     #The GENE scanlogs don't have the parameters in the same order as the sampler
        #     #So we just check if they have the same numbers regardless of the order with sort
        #     # Also the scanlogs have both omn while the sampler just has one since they are the same (to conserve quasineutrality)
        #     # This is why the unique is there, to remove the duplicated omn. 
        #     #print('EQUAL??', np.sort(np.unique(np.round(self.x[i],2))), np.sort(np.unique(np.round(sampler.samples_array[i],2))))
        #     order_bool = np.sort(np.unique(np.round(self.x[i],3))) == np.sort(np.unique(np.round(sampler.samples_array[i],3)))
        #     sample_order_bool.append(order_bool)
        # sample_order_bool = np.concatenate(sample_order_bool)
        # all_corect_order = all(sample_order_bool)
        # print("RESULT: ", all_corect_order, "\n")
        # #_____________________________________________________________
            
        # if not all_corect_order:
        #     print('The ssg_sampler.samples_array and the ssg_dataset.x have samples that are not in the same order. Thus ssg_poly cannot work.')
        #     raise KeyError
        # else:
        #     print("\n MATCHING THE DATASET SAMPLES TO THE SAMPLERS SAMPLES. THIS IS BECAUSE THEY CAN HAVE DIFFERENT COLUMN CONVENTIONS.")
        #     # Now the order is correct we can safely make them the same. 
        #     new_df = sampler.df.copy()
        #     new_df['growthrate'] = self.df['growthrate'].to_numpy()
        #     new_df['frequency'] = self.df['frequency'].to_numpy()
        #     new_df.insert(0, 'run_time', self.df['run_time'].to_numpy())
        #     self.df = new_df
        #     self.set_from_df()
        #     print("COMPLETE \n")


    def match_parameters_order(self, parameters):
        ordered_head = [] 
        for param in parameters:
            ordered_head.append(self.parameters_map[param])
        self.df = self.df.loc[:,['run_time',*ordered_head,'growthrate','frequency']]
        
    def remove_parameter(self, parameter_name):
        self.df = self.df.drop(columns=[parameter_name])
        self.set_from_df()

    def train_time_model(self, model):
        self.time_model = model
        self.time_model.train(self.x, self.run_time)

    def concat(self, datasets):
        dfs = [self.df]
        rest_df_ncol=[]
        for ds in datasets:
            rest_df_ncol.append(ds.rest_df_ncol)
            dfs.append(ds.df)
        if not all(x == rest_df_ncol[0] for x in rest_df_ncol):
            raise ValueError("Daniel says: there must be the same number of rest columns in each data set. Or else the concat won't work.")
        else:
            rest_df_ncol = rest_df_ncol[0]
        joint_df = pd.concat(dfs)
        self.df = joint_df
        self.set_from_df(df=joint_df,rest_df_ncol=rest_df_ncol)
        return self




class SSG_ScanData(ScanData):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    # def match_sampler(self, sampler):
    #     print("\nCHECKING THAT THE SSG SAMPLER AND DATASET HAVE MATCHING ORDER OF SAMPLES...")
    #     # Check that the data and sampler have the same order________
    #     sample_order_bool = []
    #     for i in range(len(self.x)):
    #         sampler.samples_array[i]
    #         self.growthrate_train[i]
    #         #print(self.x[i], sampler.samples_array[i], f'gr {self.growthrate_train[i]}')
    #         #The GENE scanlogs don't have the parameters in the same order as the sampler
    #         #So we just check if they have the same numbers regardless of the order with sort
    #         # Also the scanlogs have both omn while the sampler just has one since they are the same (to conserve quasineutrality)
    #         # This is why the unique is there, to remove the duplicated omn. 
    #         #print('EQUAL??', np.sort(np.unique(np.round(self.x[i],2))), np.sort(np.unique(np.round(sampler.samples_array[i],2))))
    #         order_bool = np.sort(np.unique(np.round(self.x[i],2))) == np.sort(np.unique(np.round(sampler.samples_array[i],2)))
    #         sample_order_bool.append(order_bool)
    #     sample_order_bool = np.concatenate(sample_order_bool)
    #     all_corect_order = all(sample_order_bool)
    #     print("RESULT: ", all_corect_order, "\n")
    #     #_____________________________________________________________
        
        # if not all_corect_order:
        #     print('The ssg_sampler.samples_array and the ssg_dataset.x have samples that are not in the same order. Thus ssg_poly cannot work.')
        #     raise KeyError
        # else:
        #     print("\n MATCHING THE DATASET SAMPLES TO THE SAMPLERS SAMPLES. THIS IS BECAUSE THEY CAN HAVE DIFFERENT COLUMN CONVENTIONS.")
        #     # Now the order is correct we can safely make them the same. 
        #     self.x = sampler.samples_array
        #     print("COMPLETE \n")

if __name__ == '__main__':
    import os
    import sys
        
    sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/gene_ml')
    from parsers.GENEparser import GENE_scan_parser
    base_params_path = os.path.join(os.getcwd(),'parameters_base_dp')
    save_dir='/project/project_462000451/gene_out/gene_auto'
    save_dir = "temp/"
    parser = GENE_scan_parser(save_dir, base_params_path, save_dir)

    # ssh_path = None
#    save_dir = "/scratch/project_462000451/gene_out/gene_auto/testing_batchscans3"'SSG_2p_l3_uq'
    save_dir = "/scratch/project_462000451/gene_out/gene_auto/nan100"
    host = 'lumi'
    #data_set = ScanData('100_3p', parser, ssh_path=ssh_path)
    data_set = ScanData('testwithnan', parser, host, save_dir=save_dir)
    

    print('HEAD',data_set.head)
    print('POINTS', data_set.x)
    print('GROWTHRATE', data_set.growthrates)
    print('FREQUENCY', data_set.frequencies)    

    print('df FIRST 5\n',data_set.df.head(5))

    
