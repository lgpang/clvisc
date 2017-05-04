#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 03 May 2017 07:20:17 AM CEST

import numpy as np
import os

def ebe_mean(path, kind='dndeta', hadron='charged', rap='Eta'):
    event_dirs = os.listdir(path)
    pid = 'charged'
    if hadron == 'pion': pid = 211
    elif hadron == 'kaon': pid = 321
    elif hadron == 'proton': pid = 2212

    fname = None
    if kind == 'dndeta':
        fname = "dNdEta_mc_%s.dat"%pid
    elif kind == 'vn':
        fname = "vn_%s.dat"%pid
    elif kind == 'dndpt':
        fname = "dN_over_2pid%sptdpt_mc_%s.dat"%(rap, pid)

    res = []
    for f in event_dirs:
        fdata = os.path.join(path, f, fname)
        if os.path.exists(fdata):
            dat = np.loadtxt(fdata)
            res.append(dat)
    return np.array(res).mean(axis=0)
