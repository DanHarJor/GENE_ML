import os
import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
try:
    from .base import DataSet
except:
    try:
        from base import DataSet
    except:
        raise ImportError 

class ScanData(DataSet):
    def __init__(self, name, parser, host=None, remote_path=None, test_percentage=50, random_state=47):
        '''
        To retrieve data from the server remote path and host should be defined
        When the string ssh <host> is entered in to he command line a ssh terminal should be started.
        The remote path needs needs to point to either a diectory that contains the scanfile* folders from a GENE scan or a specific scan.log file.
        '''
        print('Initialising dataset')
        self.name = name
        self.parser = parser
        self.host=host
        self.remote_path = remote_path

        self.test_percentage=test_percentage

        self.random_state=random_state
        ssh_path = f"{host}:{remote_path}"
        self.scan_log_path = os.path.join(os.getcwd(), 'scanlogs', self.name)
        # self.scan_log_dir = os.path.join(os.getcwd(), 'scanlogs')

        if not os.path.exists(os.path.join(os.getcwd(), 'scanlogs')):
            os.mkdir(os.path.join(os.getcwd(), 'scanlogs'))


        if not os.path.exists(self.scan_log_path): 
            print('MAKING SCANLOG DIR') 
            os.mkdir(self.scan_log_path)

        if remote_path!=None:
            self.retrieve_remote_logs(ssh_path)

        if os.path.isfile(self.scan_log_path):
            print('\nLOADING FROM SCANLOG FILE')
            self.df = self.load_from_file(self.scan_log_path)
            self.df, n_samp, n_requested, n_samp_nonan = self.remove_nans(self.df)
            
        print('\nLOADING SCANLOG/S')
        self.df, n_samp, n_requested, n_samp_nonan = self.load_from_dir(self.scan_log_path)

        print(f'\n{n_samp} SAMPLES RAN OUT OF {n_requested} BEFORE MAX WALLTIME:')
        print("NUMBER OF SAMPLES AFTER REMOVING NaN's:", n_samp_nonan)
        nan_percentage = (n_samp-n_samp_nonan)*100/n_samp
        print('NaN PERCENTAGE = ', nan_percentage)

        self.head = list(self.df.columns)
        self.x = self.df[self.head[0:-2]].to_numpy(dtype=float)
        self.growthrates = self.df['growthrate'].to_numpy(dtype=float)
        self.frequencies = self.df['frequency'].to_numpy(dtype=float)
        
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


    def load_from_file(self,data_path):
        print(f'\nLOADING SCANLOG INTO PYTHON {data_path}')
        df = self.parser.read_output_file(data_path)
        return df

    def retrieve_remote_logs(self,ssh_path):
        print('\nRETRIEVING SCANLOG/S VIA scp FROM REMOTE')
        print(f'SCANLOG PATH: {ssh_path}')
        # subprocess.run(['scp','-r',ssh_path, self.scan_log_path])
        # print('\n\nDEBUG\n\n',self.scan_log_path, type(self.scan_log_path), str(self.scan_log_path))
        if 'scan.log' in ssh_path: #Then we are only taking one file
            print('RETRIVING REMOTE FILE')
            os.system(f'scp -r {ssh_path} {self.scan_log_path}')
        else: # we should have a directory and need to take all scan logs
            print('RETRIVING FROM REMOTE DIR')
            i = 0
            results = []
            while True:
                j=0
                while True:
                    results.append(os.system(f"scp '{ssh_path}/batch*{i}/scanfiles*{j}/scan.log' {os.path.join(self.scan_log_path,f'scan_{i}_{j}.log')}"))
                    print(results)
                    j+=1
                    if results[-3:] == [256,256,256]:
                        break
                # result = os.system(f"scp 'lumi:/scratch/project_462000451/gene_out/gene_auto/testing_batchscans/scanfiles*{i}/scan.log' $PWD/scanlogs/testing_batchscans/scan{i}.log")
                i+=1
                if results[-6:] == [256,256,256,256,256,256]:
                    break 
            print('THE ABOVE ERROR IS EXPECTED')
        
        print(f'SCANLOG/S RETRIEVED AND SAVED TO:{self.scan_log_path}')
        

    def load_from_dir(self, data_path):
        '''
        This function assumes the directory only contains log files
        '''
        if not os.path.isdir(data_path): raise NotADirectoryError
        
        dfs, n_samp_all, n_requested_all, n_samp_nonan_all = [], [], [], []
        scanlog_paths = np.sort(np.array(os.listdir(data_path))) 
        for scanlog in scanlog_paths:
            df = self.load_from_file(os.path.join(data_path,scanlog))
            df, n_samp, n_requested, n_samp_nonan = self.remove_nans(df)
            dfs.append(df); n_samp_all.append(n_samp); n_requested_all.append(n_requested); n_samp_nonan_all.append(n_samp_nonan)
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
    
    def remove_parameter(self, parameter_name):
        self.df = self.df.drop(columns=[parameter_name])
        self.head = list(self.df.columns)
        self.x = self.df[self.head[0:-2]].to_numpy(dtype=float)
        self.split()

if __name__ == '__main__':
    sys.path.append('/home/djdaniel/DEEPlasma/GENE_ML/gene_ml')
    from parsers.GENEparser import GENE_scan_parser
    base_params_path = os.path.join(os.getcwd(),'parameters_base_uq')
    remote_save_dir='/project/project_462000451/gene_out/gene_auto'
    parser = GENE_scan_parser(base_params_path, remote_save_dir)

    # ssh_path = None
#    remote_path = "/scratch/project_462000451/gene_out/gene_auto/testing_batchscans3"'SSG_2p_l3_uq'
    remote_path = "/scratch/project_462000451/gene_out/gene_auto/SSG_2p_l3_uq"
    host = 'lumi'
    #data_set = ScanData('100_3p', parser, ssh_path=ssh_path)
    data_set = ScanData('testing_batchscans3', parser, host, remote_path=remote_path)
    

    print('HEAD',data_set.head)
    print('POINTS', data_set.x)
    print('GROWTHRATE', data_set.growthrates)
    print('FREQUENCY', data_set.frequencies)    

    print('df FIRST 5\n',data_set.df.head(5))

    
