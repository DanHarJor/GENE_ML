# This is a monkey patch so that IFS_scripts can work with files that are accessed via ssh

def monkey_patching_open(config):
    import builtins

    # Check if the original_open attribute exists in builtins
    if not hasattr(builtins, 'original_open'):
        # Save the original open function as an attribute of builtins
        builtins.original_open = builtins.open

        def open_file(self, file_path, mode='r'):
            if self.config.local_username in file_path:
                try: 
                    file = builtins.original_open(file_path, mode)
                except:
                    # print('using paramiko')
                    file = config.paramiko_sftp_client.open(file_path, mode)
            else:
                try:
                    # print('using paramiko 2')
                    file = config.paramiko_sftp_client.open(file_path, mode)
                except:
                    file = builtins.original_open(file_path, mode)
            return file
        
        # Replace the built-in open function with the custom one
        builtins.open = open_file