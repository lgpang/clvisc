#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 26 Nov 2015 15:03:48 CET

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style



def plot_dNdEta(num_of_events = 8):
    tau0p2 = np.loadtxt('/lustre/nyx/hyihp/lpang/pbpb5p02/event301/dN_over_2pidYptdpt_mc_charged.dat')
    eta = tau0p2[:, 0]

    dat = np.loadtxt('dNdPt_2p76.dat')
    # 1304.0347

    plt.errorbar(dat[:,0], dat[:,3], dat[:,4], fmt='o', label=r'$ALICE\ 0-5\%$')
    plt.semilogy(eta, tau0p2[:, 1], label=r'$CLVisc$')

    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$\frac{dN_{ch}}{2\pi dY p_Tdp_T}\ [GeV^{-2}]$')

    smash_style.set()
    plt.legend(loc='best', title=r'$Pb+Pb\ 2760\ GeV$')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    #plt.text(-6, 1650, '(b)')
    plt.xlim(0, 4)
    plt.ylim(1.0E-5, 1.0E4)
    plt.show()


plot_dNdEta()

