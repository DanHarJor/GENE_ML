class Parser():
    def __init__():
        NotImplemented
    
    def open_file(self, file_path, mode='r'):
        try: 
            file = open(file_path, mode)
        except:
            # print('using paramiko')
            file = self.config.paramiko_sftp_client.open(file_path, mode)
        return file
    
    def read_species_names(self, parameters_path):
        names = []
        with self.open_file(parameters_path) as parameters_file:
            for line in parameters_file:
                if 'name' in line:
                    names.append(line.split('=')[-1].strip().strip("'")) 
        #names = [s.replace('i', 'ion').replace('e', 'electron') for s in names]
        return names