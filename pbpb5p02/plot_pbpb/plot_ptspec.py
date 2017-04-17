#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 26 Nov 2015 15:03:48 CET

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style



def plot_dNdEta(num_of_events = 8):
    charged = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/dNdPt_charged_noovers.dat')
    pion = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/dNdPt_pion_noovers.dat')
    kaon = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/dNdPt_kaon_noovers.dat')
    proton = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/dNdPt_proton_noovers.dat')

    #pion = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/event5/dN_over_2pidYptdpt_mc_211.dat')
    #kaon = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/event5/dN_over_2pidYptdpt_mc_321.dat')
    #proton = np.loadtxt('/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p08/event5/dN_over_2pidYptdpt_mc_2212.dat')

    eta = pion[:, 0]

    dat0 = np.loadtxt('dNdPt_2p76.dat', skiprows=10)
    dat = np.loadtxt('dNdPt_Alice/dNdPt_pbpb2760_0_5_pion_exp.dat', skiprows=10)
    dat2 = np.loadtxt('dNdPt_Alice/dNdPt_pbpb2760_0_5_kaon_exp.dat', skiprows=10)
    dat3 = np.loadtxt('dNdPt_Alice/dNdPt_pbpb2760_0_5_proton_exp.dat', skiprows=10)
    # 1304.0347

    plt.errorbar(dat0[:,0], dat0[:,3], dat0[:,6], fmt='o', label=r'$ALICE\ charged$')
    plt.errorbar(dat[:,0], dat[:,3], dat[:,6], fmt='o', label=r'$ALICE\ \pi^+$')
    plt.errorbar(dat2[:,0], dat2[:,3], dat2[:,6], fmt='o', label=r'$ALICE\ K^+$')
    plt.errorbar(dat3[:,0], dat3[:,3], dat3[:,6], fmt='o', label=r'$ALICE\ p$')

    plt.semilogy(eta, charged[:, 1], label=r'$CLVisc\ \pi^+$')
    plt.semilogy(eta, pion[:, 1], label=r'$CLVisc\ \pi^+$')
    plt.semilogy(eta, kaon[:, 1], label=r'$CLVisc\ K^+$')
    plt.semilogy(eta, proton[:, 1], label=r'$CLVisc\ p$')
    #plt.semilogy(eta, 3.5*proton[:, 1], label=r'$CLVisc$')

    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$\frac{dN_{ch}}{2\pi dY p_Tdp_T}\ [GeV^{-2}]$')

    smash_style.set()
    plt.legend(loc='best', ncol=2, mode='expand')
    plt.title(r'$Pb+Pb\ 2.76\ TeV,\ 0-5\%$')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    #plt.text(-6, 1650, '(b)')
    plt.xlim(0, 4)
    plt.ylim(1.0E-3, 1.0E5)
    plt.savefig('dNdPt_hydro_vs_LHC.pdf')
    plt.show()


plot_dNdEta()

