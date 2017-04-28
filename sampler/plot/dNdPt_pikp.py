#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import pandas as pd
from common_plotting import smash_style

def get_dNdPt(dat0, pid=211):
    dat = dat0[dat0[:,5]==pid]
    E = dat[:,0]
    pz = dat[:,3]
    Yi = dat[:,4]
    dN, Y = np.histogram(Yi, bins=50)
    dY = (Y[1:]-Y[:-1])
    Y = 0.5*(Y[:-1]+Y[1:])
    NEvent = 2000
    pti = np.sqrt(dat[:,1]**2+dat[:,2]**2)

    pti = pti[np.abs(Yi)<0.8] 

    dN, pt = np.histogram(pti, bins=50)

    dpt = pt[1:]-pt[:-1]
    pt = 0.5*(pt[1:]+pt[:-1])

    return pt, dN/(2*np.pi*float(NEvent)*dpt*1.6)




if __name__=='__main__':
    fname='event_ideal/mc_particle_list.dat'
    dat = pd.read_csv(fname, skiprows=1, header=None, \
            sep=' ', dtype = float).values
    pt0, dNdPt_0 = get_dNdPt(dat, pid=211)
    pt1, dNdPt_1 = get_dNdPt(dat, pid=321)
    pt2, dNdPt_2 = get_dNdPt(dat, pid=2212)

    fig, ax = plt.subplots(1,1)
    dndpt0 = np.loadtxt('event_ideal/dNdYPtdPt_over_2pi_Reso211.dat')
    dndpt1 = np.loadtxt('event_ideal/dNdYPtdPt_over_2pi_Reso321.dat')
    dndpt2 = np.loadtxt('event_ideal/dNdYPtdPt_over_2pi_Reso2212.dat')
    ax.plot(dndpt0[:, 0], dndpt0[:,0]*dndpt0[:, 1], 'r-', label='smooth')
    ax.plot(dndpt1[:, 0], dndpt1[:,0]*dndpt1[:, 1], 'r-', label='smooth')
    ax.plot(dndpt2[:, 0], dndpt2[:,0]*dndpt2[:, 1], 'r-', label='smooth')

    ax.plot(pt0, dNdPt_0, 'bo-', ms=5, label='mc pion+')
    ax.plot(pt1, dNdPt_1, 'bs-', ms=5, label='mc kaon+')
    ax.plot(pt2, dNdPt_2, 'bd-', ms=5, label='mc proton')

    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel(r'$\frac{dN}{2\pi dY dp_T}\ [GeV]^{-2}\ for\ \pi^+$')

    smash_style.set()
    ax.legend(loc='best')
    plt.subplots_adjust(left=0.20, bottom=0.15, top=0.94, right=0.94)
    plt.show()
