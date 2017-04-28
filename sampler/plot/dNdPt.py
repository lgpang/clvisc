#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import pandas as pd
from common_plotting import smash_style

def get(fname, kind='pion'):
    dat = pd.read_csv(fname, skiprows=1, header=None, \
            sep=' ', dtype = float).values
    E = dat[:,0]
    pz = dat[:,3]
    Yi = dat[:,4]
    dN, Y = np.histogram(Yi, bins=50)
    dY = (Y[1:]-Y[:-1])
    Y = 0.5*(Y[:-1]+Y[1:])
    NEvent = 1000
    pti = np.sqrt(dat[:,1]**2+dat[:,2]**2)

    pti = pti[np.abs(Yi)<0.8] 

    dN, pt = np.histogram(pti, bins=50)

    dpt = pt[1:]-pt[:-1]
    pt = 0.5*(pt[1:]+pt[:-1])

    return pt, dN/(2*np.pi*float(NEvent)*dpt*1.6)




if __name__=='__main__':
    pt0, dNdPt_nodecay = get(fname='build/pmag_for_pion.dat')
    pt1, dNdPt_2decay = get(fname='build/pmag_2_body_decay.dat')
    pt2, dNdPt_23decay = get(fname='build/pmag_2_and_3_body_decay.dat')

    fig, ax = plt.subplots(1,1)
    dndpt = np.loadtxt('event_ideal/dNdYPtdPt_over_2pi_211.dat')
    ax.plot(dndpt[:, 0], dndpt[:,0]*dndpt[:, 1], 'r-', label='smooth')

    dndpt_R = np.loadtxt('event_ideal/dNdYPtdPt_over_2pi_Reso211.dat')
    ax.plot(dndpt_R[:, 0], dndpt_R[:,0]*dndpt_R[:, 1], 'g--', label='smooth after decay')

    ax.plot(pt0, dNdPt_nodecay, 'bo-', ms=5, label='mc no')
    ax.plot(pt1, dNdPt_2decay, 'bs-', ms=5, label='mc 2-body')
    ax.plot(pt2, dNdPt_23decay, 'bd-', ms=5, label='mc 23-body')

    ax.set_xlabel(r'$p_T$ [GeV]')
    ax.set_ylabel(r'$\frac{dN}{2\pi dY dp_T}\ [GeV]^{-2}\ for\ \pi^+$')

    smash_style.set()
    ax.legend(loc='best')
    plt.subplots_adjust(left=0.20, bottom=0.15, top=0.94, right=0.94)
    plt.show()
