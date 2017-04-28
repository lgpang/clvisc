#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import pandas as pd
from common_plotting import smash_style

def get_dNdY(dat, pid=211):
    #dat = np.loadtxt( 'build/pmag_for_pion.dat', skiprows=1)
    E = dat[:,0]
    pz = dat[:,3]
    Yi = dat[:,4]
    dN, Y = np.histogram(Yi[dat[:, 5]==pid], bins=50)
    dY = (Y[1:]-Y[:-1])
    Y = 0.5*(Y[:-1]+Y[1:])
    NEvent = 2000
    return Y, dN/(dY*float(NEvent))



if __name__=='__main__':
    #fname='plot/pmag_23body_decay.dat'
    fname='event_ideal/mc_particle_list.dat'
    dat = pd.read_csv(fname, skiprows=1, header=None, \
            sep=' ', dtype = float).values

    Y2, dNdY_23decay = get_dNdY(dat)

    Y2_proton, dNdY_23decay_proton = get_dNdY(dat, pid=2212)
    Y2_kaon, dNdY_23decay_kaon = get_dNdY(dat, pid=321)

    fig, ax = plt.subplots(1,1)
    dndy_smooth_R = np.loadtxt('event_ideal/dNdY_Reso211.dat')
    #sm0, = ax.plot(dndy_smooth_R[:, 0], dndy_smooth_R[:,1], 'm-', label='smooth 234body')

    dndy_smooth_23body = np.loadtxt('event_ideal/ResoDecayTest/ResoDecay_23body/dNdY_Reso211.dat')
    sm1, = ax.plot(dndy_smooth_23body[:, 0], dndy_smooth_23body[:,1], 'r-', label='smooth 23body')

    #dndy_smooth_2body = np.loadtxt('event_ideal/ResoDecayTest/ResoDecay_2body/dNdY_Reso211.dat')
    #sm2, = ax.plot(dndy_smooth_2body[:, 0], dndy_smooth_2body[:,1], 'b-', label='smooth 2body')

    #dndy_smooth = np.loadtxt('event_ideal/dNdY_211.dat')
    #sm3, = ax.plot(dndy_smooth[:, 0], dndy_smooth[:,1], 'g-', label='smooth no decay')

    dndy_smooth_23body_k = np.loadtxt('event_ideal/ResoDecayTest/ResoDecay_23body/dNdY_Reso321.dat')
    sm4, = ax.plot(dndy_smooth_23body_k[:, 0], dndy_smooth_23body_k[:,1], 'b-', label='smooth 23body')

    dndy_smooth_23body_p = np.loadtxt('event_ideal/ResoDecayTest/ResoDecay_23body/dNdY_Reso2212.dat')
    sm5, = ax.plot(dndy_smooth_23body_p[:, 0], dndy_smooth_23body_p[:,1], 'g-', label='smooth 23body')

    mc0, = ax.plot(Y2, dNdY_23decay, 'ro', label='mc 23-body')
    #mc1, = ax.plot(Y1, dNdY_2decay, 'rs', label='mc 2-body')
    #mc2, = ax.plot(Y0, dNdY_nodecay, 'ro', label='mc no decay')
    #mc_scale, = ax.plot(Y0, 1.79159*dNdY_nodecay, 'r--', label='1.79159 * mc no decay')

    mc3, = ax.plot(Y2_kaon, dNdY_23decay_kaon, 'rs', label='mc kaon')
    mc4, = ax.plot(Y2_proton, dNdY_23decay_proton, 'r^', label='mc proton')

    ax.set_xlabel('Y')
    ax.set_ylabel(r'$dN/dY$')

    smash_style.set()
    ax.legend(loc='best')

    legend1 = plt.legend([sm1, sm4, sm5], ['23-body pion+', '23-body kaon+', '23-body proton'],
               title='smooth', loc='upper left')

    plt.legend([ mc0, mc3, mc4], ['23-body pion+', '23-body kaon+', '23-body proton'],
               title='sampling', loc='upper right')

    plt.xlim(-10, 10)

    plt.gca().add_artist(legend1)
    plt.subplots_adjust(left=0.1, bottom=0.15, top=0.96, right=0.96)
    plt.show()
