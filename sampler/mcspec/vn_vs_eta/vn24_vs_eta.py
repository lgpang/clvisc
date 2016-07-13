#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 07 Jul 2016 02:13:05 AM CEST

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style

import pyopencl as cl

if __name__=='__main__':
    vn = np.empty((200, 20, 7))
    vn[:] = np.NAN

    for eid in range(0, 200):
        try:
            vn_new = np.loadtxt('event%d/vn24_vs_eta.txt'%eid)
            vn[eid] = vn_new
            print(eid, ' finished')
        except:
            print("no data")

    vn_mean = np.nanmean(vn, axis=0)

    #exp2 = np.loadtxt('v2_vs_eta_exp.dat')
    #exp3 = np.loadtxt('v3_vs_eta_exp.dat')
    #exp4 = np.loadtxt('v4_vs_eta_exp.dat')

    exp = np.loadtxt('exp_data.dat', skiprows=10)


    plt.plot(vn_mean[:, 0], vn_mean[:, 1], 'r-', label=r'CLVisc, $\eta/s=0.16$')
    plt.plot(vn_mean[:, 0], vn_mean[:, 2], 'b--', label=r'CLVisc, $\eta/s=0.16$')
    #plt.plot(vn_mean[:, 0], vn_mean[:, 3], 'g:', label=r'CLVisc, n=4, $\eta/s=0.08$')

    plt.errorbar(exp[:, 0], exp[:, 1], yerr=exp[:, 3], elinewidth=20, ls='', color='r', label=r'ALICE, n=2')
    plt.errorbar(exp[:, 0], exp[:, 7], yerr=exp[:, 9], elinewidth=20, ls='', color='b', label=r'ALICE, n=3')
    #plt.errorbar(exp[:, 0], exp[:, 10], yerr=exp[:, 11], label=r'ALICE, n=4')

    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$v_n\{2\}$')
    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV,\ 0-5\%$')
    plt.ylim(0.0, 0.04)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    smash_style.set()

    plt.legend(ncol=2, loc="upper left", bbox_to_anchor=[0, 1])
    plt.savefig('vn2_vs_eta_0_5.pdf')
    plt.show()



