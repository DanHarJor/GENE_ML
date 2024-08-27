def sec_to_time_format(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60, )
    d, h = divmod(h, 24)
    s,m,h,d = str(int(s)), str(int(m)), str(int(h)), str(int(d))
    if len(d)==1: d = '0'+d
    if len(h)==1: h = '0'+h
    if len(m)==1: m = '0'+m
    if len(d)==1: s = '0'+s
    return f"{d}-{h}:{m}:{s}"