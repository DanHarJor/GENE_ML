import numpy as np
from .parser_base import Parser
class ParserTimeseries(Parser):
    # def __init__(self):
    #     NotImplemented

    def Qem_history(self, nrg_path, species_names=None):
        time = []
        species_number=1
        Qem = {}
        with self.open_file(nrg_path, 'r') as nrg_file:
            # do loop once to initiate Qes dict
            for line in nrg_file:
                row = line.split(' ')
                row = [item for item in row if item != '']
                if len(row)==1:
                    time.append(row[0])
                    species_number=1
                else:
                    if len(time)==1:
                        Qem[str(species_number)] = [row[7]]
                    else:
                        Qem[str(species_number)].append(row[7])
                    species_number += 1
        for i in range(1,species_number):
            Qem[str(i)] = np.array(Qem[str(i)]).astype('float')
        
        return Qem, np.array(time).astype('float')  


    def Qes_history(self, nrg_path, species_names=None):
        time = []
        species_number=1
        Qes = {}
        with self.open_file(nrg_path, 'r') as nrg_file:
            # do loop once to initiate Qes dict
            for line in nrg_file:
                row = line.split(' ')
                row = [item for item in row if item != '']
                if len(row)==1:
                    time.append(row[0])
                    species_number=1
                else:
                    if len(time)==1:
                        Qes[str(species_number)] = [row[6]]
                    else:
                        Qes[str(species_number)].append(row[6])
                    species_number += 1
        for i in range(1,species_number):
            Qes[str(i)] = np.array(Qes[str(i)]).astype('float')
        
        return Qes, np.array(time).astype('float')  

    def growthrate_history(self, energy_path):
        # Uses the instantanious grothrate from the energy file. Info by tobias in email
        with self.open_file(energy_path, 'r') as energy_file:
            time = []
            growthrate = []
            for line in energy_file:
                row = line.split(' ')
                row = [item for item in row if item != '']
                # row = np.array(row).astype('float')
                if len(row)== 14:
                    row = np.array(row).astype('float')
                    growthrate.append(0.5*(row[3]/row[2]))
                    time.append(row[0])
        return np.array(growthrate), np.array(time)