#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 26 Nov 2015 15:03:48 CET

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style


def plot_hydro(directory):
    tau0p2 = np.loadtxt('/lustre/nyx/hyihp/lpang/pbpb5p02/auau200/%s/dNdEta_mc_charged.dat'%directory)
    eta = tau0p2[:, 0]
    if directory == 'event0_6':
        plt.plot(eta, tau0p2[:, 1], 'k-', label=r'$CLVisc$')
    else:
        plt.plot(eta, tau0p2[:, 1], 'k-')



def plot_dNdEta(num_of_events = 8):
    plot_hydro('event0_6')
    plot_hydro('event6_15')
    plot_hydro('event15_25')
    plot_hydro('event25_35')

    #ebe = np.loadtxt('dNdEta_0_6_ebe.dat')

    #plt.plot(ebe[:, 0], ebe[:, 1], 'r--')

    # 1304.0347
    dat0 = np.loadtxt('dNdEta_0_6.dat')
    dat1 = np.loadtxt('dNdEta_6_15.dat')
    dat2 = np.loadtxt('dNdEta_15_25.dat')
    dat3 = np.loadtxt('dNdEta_25_35.dat')

    plt.errorbar(dat0[:,0], dat0[:,2], yerr=[-dat0[:,4], dat0[:, 3]], fmt='ro', label=r'$PHOBOS$')
    plt.errorbar(dat1[:,0], dat1[:,2], yerr=[-dat1[:,4], dat1[:, 3]], fmt='bo')
    plt.errorbar(dat2[:,0], dat2[:,2], yerr=[-dat2[:,4], dat2[:, 3]], fmt='go')
    plt.errorbar(dat3[:,0], dat3[:,2], yerr=[-dat3[:,4], dat3[:, 3]], fmt='mo')

    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\frac{dN_{ch}}{d\eta}$')

    xcod = [-0.7, -0.7, -0.7, -0.7]
    ycod = [720, 550, 400, 260]
    text = ['0-6%', '6-15%', '15-25%', '25-35%']

    for i in range(4):
        plt.text(xcod[i], ycod[i], text[i], size=20)

    smash_style.set(line_styles=False)
    plt.legend(loc='best', title=r'$Au+Au\ 200\ GeV$')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    plt.xlim(-8, 8)
    plt.ylim(0, 1000)
    plt.savefig('dNdEta_AuAu200.pdf')
    plt.show()


plot_dNdEta()

