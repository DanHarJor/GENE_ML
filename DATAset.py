import os
import sys
import subprocess
import pandas as pd

class DATAset():
    def __init__(self, name):
        print('Initialising dataset')
        self.name = name
        self.scan_log_path = os.path.join(os.getcwd(), 'scanlogs', f'{self.name}_scan.log')
        self.df = None

    def load_from_file(self,data_path):
        print(f'Loading Data From {data_path}')
        self.parse_scan_log(self.scan_log_path)
        return self.df

    def load_from_remote(self,ssh_path):
        subprocess.run(['scp',ssh_path, os.path.join(os.getcwd(),'scanlogs')])
        subprocess.run(['mv','scanlogs/scan.log',f'scanlogs/{self.name}_scan.log'])
        self.load_from_file(os.path.join(os.getcwd(),'scanlogs',f'{self.name}_scan.log'))
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
        print(head)

    

        self.df = pd.read_csv(scan_log_path, sep='|',skiprows=1, names=head)
        for i in range(len(self.df)):
            split = self.df[head[-1]][i].lstrip().rstrip().split(' ')      
            growthrate.append(split[0])
            frequency.append(split[-1])
        self.df['growthrate'] = growthrate
        self.df['frequency'] = frequency
        self.df = self.df.drop(columns=[head[0],head[-1]])
        self.df = self.df
        return self.df

if __name__ == '__main__':
    data_set = DATAset('100_3p')
    df = data_set.load_from_remote('lumi:/scratch/project_462000451/gene_out/ped2_safescan/scanfiles0012/scan.log')
    print(df)
