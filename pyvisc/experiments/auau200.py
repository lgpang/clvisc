#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 26 Nov 2015 15:03:48 CET

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style
from helper import ebe_mean
import os
import pandas as pd

def dndeta(cent='0_6'):
    '''return dndeta, where eta=dndeta[:, 0] and dNdEta=dndeta[:,1]'''
    return np.loadtxt('data/auau200/dNdEta_%s.dat'%cent)

def ptspec(hadron='pion', cent='0_5'):
    '''return pt, (1/2pi)dN/dYptdpt [GeV^{-2}] '''
    clist1 = ['minbias', '0_5', '5_10', '10_15', '15_20', '20_30', '30_40', '40_50']
    clist2 = ['50_60', '60_70', '70_80', '80_92']
    dat = None
    if cent in clist1:
        dat = pd.read_csv("data/auau200/ptspec/%s1.csv"%hadron)
    elif cent in clist2:
        dat = pd.read_csv("data/auau200/ptspec/%s2.csv"%hadron)
    cent = cent.replace('_', '-')
    err = dat['%serr'%cent].values
    return dat['pT[GeV/c]'].values, dat[cent].values, err, err


def cmp_dndeta(path_to_results='', cent = ['0_6', '6_15', '15_25', '25_35']):
    '''compare ebe-mean dndeta with PHOBOS exp data'''
    # 1304.0347
    for i, c in enumerate(cent):
        if i == 0:
            label0 = r'$PHOBOS$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None
        dat = dndeta(c)
        plt.errorbar(dat[:, 0], dat[:, 2], yerr=(-dat[:, 4], dat[:, 3]), label=label0, color='r')
        path = os.path.join(path_to_results, c)
        hydro = ebe_mean(path)
        plt.plot(hydro[:, 0], hydro[:, 1], color='k', label=label1)
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



def cmp_ptspec(path_to_results='', cent = ['0_5', '10_15', '20_30', '30_40'], hadron='pion'):
    for i, c in enumerate(cent):
        if c == '0_5':
            label0 = r'$PHENIX$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None

        x, y, yerr0, yerr1 = ptspec(hadron, c)
        shift = 5**(-i)
        plt.errorbar(x, y*shift, yerr=(yerr0*shift, yerr1*shift), label=label0, fmt='ro')
        path = os.path.join(path_to_results, c.replace('-', '_'))
        dndpt = ebe_mean(path, kind='dndpt', hadron=hadron, rap='Y')
        plt.semilogy(dndpt[:, 0], dndpt[:, 1]*shift, 'k-', label=label1)
        k, ang = 1.3, 25
        if hadron == 'kaon':
            k, ang = 2.0, 20
        elif hadron == 'proton':
            k, ang = 2.5, 15
        plt.text(x[8], shift*y[8]*k, c.replace('_', '-'), rotation=-ang, size=20)
        plt.text(x[13], shift*y[13]*k, r'$\times\ %s$'%(1/float(5**i)), rotation=-ang, size=20)
    plt.xlim(0, 3)
    plt.ylim(1.0E-5, 1.0E3)
    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$(1/2\pi)d^2 N/dYp_Tdp_T\ [GeV]^{-2}$')
    if hadron == 'pion':
        plt.title(r'$Au+Au\ \sqrt{s_{NN}}=200\ GeV,\ for\ \pi^+$', fontsize=30)
    elif hadron == 'kaon':
        plt.title(r'$Au+Au\ \sqrt{s_{NN}}=200\ GeV,\ for\ K^+$', fontsize=30)
    elif hadron == 'proton':
        plt.title(r'$Au+Au\ \sqrt{s_{NN}}=200\ GeV,\ for\ proton$', fontsize=30)
    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig('figs/auau200_ptspec_%s.pdf'%hadron)
    plt.show()



if __name__ == '__main__':
    path = "/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results/"
    #cmp_dndeta(path)
    cmp_ptspec(path, hadron='pion')
    cmp_ptspec(path, hadron='kaon')
    cmp_ptspec(path, hadron='proton')
