#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 26 Nov 2015 15:03:48 CET

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style


def plot_hydro(directory):
    tau0p2 = np.loadtxt('../pbpb5020/%s/dNdEta_mc_charged.dat'%directory)
    eta = tau0p2[:, 0]
    if directory == 'cent0_5':
        plt.plot(eta, tau0p2[:, 1], 'k-', label=r'$CLVisc\ Pb+Pb\ \sqrt{s_{NN}}=5.02\ TeV$')
    else:
        plt.plot(eta, tau0p2[:, 1], 'k-')



def plot_dNdEta(num_of_events = 8):
    plot_hydro('cent0_5')
    plot_hydro('cent5_10')
    plot_hydro('cent10_20')
    plot_hydro('cent20_30')

    # 1304.0347
    dat0 = np.loadtxt('dNdEta_2p76.dat')

    plt.errorbar(dat0[:,0], dat0[:,3], dat0[:,4], fmt='ro', label=r'$ALICE\ Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV\ 0-5\%$')

    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\frac{dN_{ch}}{d\eta}$')

    xcod = [-0.7, -0.7, -0.7, -0.7]
    ycod = [2000, 1700, 1350, 950]
    text = ['0-5%', '5-10%', '10-20%', '20-30%']

    for i in range(4):
        plt.text(xcod[i], ycod[i], text[i], size=20)

    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    plt.xlim(-8, 8)
    plt.ylim(0, 2800)
    plt.savefig('dNdEta_pbpb5020.pdf')
    plt.show()


plot_dNdEta()

