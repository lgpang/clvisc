#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sun 30 Apr 2017 04:20:10 AM CEST

import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from common_plotting import smash_style
from helper import ebe_mean
import pandas as pd

def dndeta(cent='0-5'):
    dat = pd.read_csv('data/pbpb5020_dndeta_mideta.csv', comment='#')
    cent_mid = dat['cent_mid'].values
    cent_low = dat['cent_low'].values
    cent_high = dat['cent_high'].values
    dndeta_mid = dat['dndeta_mid']
    err_high = dat['sys+']
    err_low = -dat['sys-']
    table = {}
    for i in xrange(len(cent_low)):
        c0, c1 = cent_low[i], cent_high[i]
        trunc = lambda c: int(c) if c - int(c) < 0.01 else c
        c0, c1 = trunc(c0), trunc(c1)

        key = '%s-%s'%(c0, c1)
        table[key] = {'cent_mid':cent_mid[i], 'x':0.0, 'y':dndeta_mid[i], 'yerr0':err_low[i], 'yerr1':err_high[i]}
    # add 0-5 and 5-10 by interpolation
    table['0-5'] = {'cent_mid':2.5, 'x':0.0, 'y':0.5*(table['0-2.5']['y']+table['2.5-5']['y']),
                    'yerr0':0.5*(table['0-2.5']['yerr0']+table['2.5-5']['yerr0']),
                    'yerr1':0.5*(table['0-2.5']['yerr1']+table['2.5-5']['yerr1'])}

    table['5-10'] = {'cent_mid':7.5, 'x':0.0, 'y':0.5*(table['5-7.5']['y']+table['7.5-10']['y']),
                    'yerr0':0.5*(table['5-7.5']['yerr0']+table['7.5-10']['yerr0']),
                    'yerr1':0.5*(table['5-7.5']['yerr1']+table['7.5-10']['yerr1'])}
    return table




def cmp_dndeta(path_to_results='', cent = ['0-5', '5-10', '10-20', '20-30']):
    xpos = [-0.5, -1.0, -1, -1]
    table = dndeta()
    for i, c in enumerate(cent):
        if c == '0-5':
            label0 = r'$ALICE$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None
        plt.errorbar([table[c]['x']], [table[c]['y']], yerr=([table[c]['yerr0']], [table[c]['yerr1']]), label=label0, color='r') 
        path = os.path.join(path_to_results, c.replace('-', '_'))
        hydro = ebe_mean(path)
        plt.plot(hydro[:, 0], hydro[:, 1], color='k', label=label1)
        neta = len(hydro[:, 0])
        plt.text(xpos[i], hydro[neta/2, 1]+100, c, fontsize=25)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$dN_{ch}/d\eta$')
    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.ylim(0, 2500)
    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=5.02\ TeV$', fontsize=30)
    plt.savefig('pbpb5020_dndeta.pdf')
    plt.show()

def cmp_ptspec(path_to_results='', cent = ['0-5', '5-10', '10-20'], hadron='pion'):
    #from pbpb2760 import dNdPt
    #exp = dNdPt()
    for i, c in enumerate(cent):
        if i == 0:
            label0 = r'$ALICE$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None

        shift = 5**(-i)
        #x, y, yerr0, yerr1 = exp.get(hadron, c)

        #plt.errorbar(x, y*shift, yerr=(yerr0*shift, yerr1*shift), label=label0, fmt='o', color='r')
        path = os.path.join(path_to_results, c.replace('-', '_'))
        dndpt = ebe_mean(path, kind='dndpt', hadron=hadron, rap='Y')
        plt.semilogy(dndpt[:, 0], 2*dndpt[:, 1]*shift, label=label1, color='k')
        ytxt, theta = 2.5, -20
        if hadron == 'kaon':
            ytxt, theta = 1.8, -12
        elif hadron == 'proton':
            ytxt, theta = 2.0, -5
        plt.text(dndpt[5, 0], ytxt*dndpt[5, 1]*shift, r'$%s$'%c, rotation=theta, size=25)
        plt.text(dndpt[12, 0], ytxt*dndpt[12, 1]*shift, r'$\times 5^{%s}$'%(-i), rotation=theta, size=25)

    plt.xlim(0, 3)
    plt.ylim(1.0E-7, 1.0E4)
    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$(1/2\pi)d^2 N/dYp_Tdp_T\ [GeV]^{-2}$')
    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.tight_layout()

    if hadron == 'pion':
        plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=5.02\ TeV,\ \pi^++\pi^-$', fontsize=30)
    elif hadron == 'kaon':
        plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=5.02\ TeV,\ K^++K^-$', fontsize=30)
    elif hadron == 'proton':
        plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=5.02\ TeV,\ p+\bar{p}$', fontsize=30)
    plt.savefig('pbpb5020_ptspec_%s.pdf'%hadron)
    plt.show()

def cmp_v2_pion(path_to_results):
    from pbpb2760 import V2
    exp = V2()
    cent = ['0-5', '5-10', '10-20', '20-30']
    for c in cent:
        if c == '0-5':
            label0 = r'$ALICE$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None
        pt, vn, yerr0, yerr1 = exp.get_ptdiff('pion', c)
        plt.errorbar(pt, vn, yerr=(yerr0, yerr1), label=label0, color='r')

        path = os.path.join(path_to_results, c.replace('-', '_'))
        vn_clvisc = ebe_mean(path, kind='vn', hadron='pion')
        plt.plot(vn_clvisc[:, 0], vn_clvisc[:, 2], label = label1, color = 'k')

        plt.text(2, vn[9]+0.01, c, fontsize=25)

    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$v_2$')
    smash_style.set(line_styles=False)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.xlim(0, 2.5)
    plt.ylim(0.001, 0.3)
    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=5.02\ TeV$', fontsize=30)
    plt.savefig('pbpb5020_pionv2.pdf')
    plt.show()





if __name__=='__main__':
    '''dndeta is not sensitive to Tfrz
    pt_spectra fit best with Tfrz=100 MeV; while v2 fits best with Tfrz=137 MeV'''
    path = "/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results"
    #cmp_dndeta(path)
    #cmp_ptspec(path, cent = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60'], hadron='pion')
    cmp_v2_pion(path)
