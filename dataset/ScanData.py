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
    def __init__(self, name, ssh_path=None, test_percentage=50, random_state=47):
        print('Initialising dataset')
        self.name = name
        self.scan_log_path = os.path.join(os.getcwd(), 'scanlogs', f'{self.name}_scan.log')
        self.df = None
        self.head = None
        self.x = None
        self.growthrates = None
        self.frequencies = None
        if ssh_path==None:
            self.load_from_file(self.scan_log_path)
        else:
            self.load_from_remote(ssh_path) 
        print(f'RANDOMLY SPLITTING DATA INTO TEST AND TRAINING SETS: {test_percentage}% test, {100-test_percentage} training.')
        self.x_train, self.x_test, self.growthrate_train, self.growthrate_test, self.frequencies_train, self.frequencies_test = train_test_split(self.x, self.growthrates, self.frequencies, test_size=test_percentage/100, random_state=random_state)

    def load_from_file(self,data_path):
        print(f'LOADING SCAN LOG INTO PYTHON {data_path}')
        self.parse_scan_log(self.scan_log_path)
        return self.df

    def load_from_remote(self,ssh_path):
        print('RETRIEVING SCANLOG VIA scp FROM REMOTE')
        print(f'scp, scanlog path: {ssh_path}')
        subprocess.run(['scp',ssh_path, os.path.join(os.getcwd(),'scanlogs')])
        subprocess.run(['mv','scanlogs/scan.log',self.scan_log_path])
        print(f'FILE RETRIEVED AND SAVED TO:{self.scan_log_path}')
        self.load_from_file(self.scan_log_path)
        return self.df
    
    def parse_scan_log(self, scan_log_path=os.path.join('/scratch/project_462000451/daniel/AUGUQ/scanfiles0002/scan.log')):
        growthrate = []
        frequency = []
        head = open(scan_log_path, 'r').readline()
        # head.replace('\n','')
        head = head.split('|')
        last_two = head[-1].split('/')
        del head[-1]
        head = head + last_two
        
        self.df = pd.read_csv(scan_log_path, sep='|',skiprows=1, names=head)
        for i in range(len(self.df)):
            split = self.df[head[-1]][i].lstrip().rstrip().split(' ')      
            growthrate.append(split[0])
            frequency.append(split[-1])
        self.df['growthrate'] = growthrate
        self.df['frequency'] = frequency
        self.df = self.df.drop(columns=[head[0],head[-1]])
        
        #removing NAN's
        nan_mask = ~np.isnan(self.df['growthrate'].to_numpy(dtype=float))
        print('NaN percentage = ',np.sum(nan_mask)*100/len(nan_mask))
        self.df = self.df.loc[nan_mask]

        self.df = self.df
        self.head = list(self.df.columns)
        self.x = self.df[self.head[0:-2]].to_numpy(dtype=float)
        self.growthrates = self.df['growthrate'].to_numpy(dtype=float)
        self.frequencies = self.df['frequency'].to_numpy(dtype=float)
        return self.df

if __name__ == '__main__':
    data_set = ScanData(name='100_3p')
    #df = data_set.load_from_remote('lumi:/scratch/project_462000451/gene_out/ped2_safescan/scanfiles0012/scan.log')
    df = data_set.load_from_file(os.path.join('scanlogs','100_3p_scan.log'))

    print('HEAD',data_set.head)
    print('POINTS', data_set.x)
    print('GROWTHRATE', data_set.growthrate)
    print('FREQUENCY', data_set.frequency)    

    print(df)

    
