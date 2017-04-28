#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import pandas as pd
from common_plotting import smash_style

def get_v2(dat, pid=211):
    NEvent = 1000
    E = dat[:,0]
    px = dat[:,1]
    py = dat[:,2]
    pz = dat[:,3]
    Yi = dat[:,4]
    pti = np.sqrt(dat[:,1]**2+dat[:,2]**2)

    phi = np.arctan2(py, px)

    phi = phi[np.abs(Yi)<0.8] 

    dN, dphi = np.histogram(phi, bins=50)

    phic = 0.5*(dphi[1:]+dphi[:-1])
    dphi = dphi[1:] - dphi[:-1]

    return phic, dN/dphi/NEvent


fname='~/Documents/D0/data/D0/mc_particle_list.dat'
dat = pd.read_csv(fname, skiprows=1, header=None, \
            sep=' ', dtype = float).values

phi, dN = get_v2(dat)
plt.plot(phi, dN)
plt.show()




