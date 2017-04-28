#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import pandas as pd
from common_plotting import smash_style

def get_dNdY(fname):
    #dat = np.loadtxt( 'build/pmag_for_pion.dat', skiprows=1)
    dat = pd.read_csv(fname, skiprows=1, header=None, \
            sep=' ', dtype = float).values
    E = dat[:,0]
    px = dat[:,1]
    py = dat[:,2]
    pz = dat[:,3]
    Yi = dat[:,4]
    p = np.sqrt(px*px + py*py + pz*pz)
    eta = 0.5*np.log((p+pz)/(p-pz))
    dN, Y = np.histogram(eta, bins=50)
    dY = (Y[1:]-Y[:-1])
    Y = 0.5*(Y[:-1]+Y[1:])
    NEvent = 2000
    return Y, dN/(dY*float(NEvent))



if __name__=='__main__':
    Y2, dNdY_23decay = get_dNdY(fname='~/data/auau200_results/ideal/cent20_25_etas0p00/cent20_25_event0/mc_particle_list.dat')

    fig, ax = plt.subplots(1,1)
    #dndy_smooth_R = np.loadtxt('event_ideal/dNdEta_Charged.dat')
    #sm0, = ax.plot(dndy_smooth_R[:, 0], dndy_smooth_R[:,1], 'm-', label='smooth 234body')

    mc0, = ax.plot(Y2, dNdY_23decay, 'rd', label='mc 23-body')

    ax.set_xlabel('Y')
    ax.set_ylabel(r'$dN/dY\ for\ charged$')

    smash_style.set()
    ax.legend(loc='best')

    plt.subplots_adjust(left=0.1, bottom=0.15, top=0.96, right=0.96)
    plt.show()
