#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 26 Nov 2015 15:03:48 CET

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style
from helper import ebe_mean
import os

def cmp_dndeta(path_to_results=''):
    # 1304.0347
    cent = ['0_6', '6_15', '15_25', '25_35']
    for i, c in enumerate(cent):
        if c == '0_6':
            label0 = r'$PHOBOS$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None
        dat = np.loadtxt('data/auau200/dNdEta_%s.dat'%c)
        plt.errorbar(dat[:, 0], dat[:, 2], yerr=(-dat[:, 4], dat[:, 3]), label=label0, color='r')
        path = os.path.join(path_to_results, c)
        dndeta = ebe_mean(path)
        plt.plot(dndeta[:, 0], dndeta[:, 1], color='k', label=label1)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$dN_{ch}/d\eta$')
    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.xlim(-8, 8)
    plt.ylim(0, 1000)
    plt.title(r'$Au+Au\ \sqrt{s_{NN}}=200\ GeV$', fontsize=30)
    xcod = [-0.4, -0.5, -0.7, -0.7]
    ycod = [720, 550, 400, 260]
    text = ['0-6', '6-15', '15-25', '25-35']

    for i in range(4):
        plt.text(xcod[i], ycod[i], text[i], size=20)

    plt.savefig('figs/auau200_dndeta.pdf')
    plt.show()

if __name__ == '__main__':
    path = "/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results/"
    cmp_dndeta(path)
