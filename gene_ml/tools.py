import pickle
def sec_to_time_format(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60, )
    d, h = divmod(h, 24)
    s,m,h,d = str(int(s)), str(int(m)), str(int(h)), str(int(d))
    if len(d)==1: d = '0'+d
    if len(h)==1: h = '0'+h
    if len(m)==1: m = '0'+m
    if len(s)==1: s = '0'+s
    return f"{d}-{h}:{m}:{s}"

def save_pkl(path, var):
    with open(path, 'wb') as pickle_file:
        pickle.dump(var, pickle_file)

def load_pkl(path):
    with open(path, 'rb') as pickle_file:
        return pickle.load(pickle_file)
    