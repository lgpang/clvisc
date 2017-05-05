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


def cmp_dndeta(path_to_results=''):
    from pbpb2760 import dNdEta
    exp = dNdEta()
    cent = ['0-5', '5-10', '10-20', '20-30']
    xpos = [-0.5, -1.0, -1, -1]
    for i, c in enumerate(cent):
        if c == '0-5':
            label0 = r'$ALICE$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None
        plt.errorbar(exp.x, exp.y[c], yerr=(exp.yerr[c], exp.yerr[c]), label=label0, color='r') 
        path = os.path.join(path_to_results, c.replace('-', '_'))
        dndeta = ebe_mean(path)
        plt.plot(dndeta[:, 0], dndeta[:, 1], color='k', label=label1)
        neta = len(dndeta[:, 0])
        plt.text(xpos[i], dndeta[neta/2, 1]+100, c, fontsize=25)
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$dN_{ch}/d\eta$')
    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.xlim(-10, 10)
    plt.ylim(0, 2000)
    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV$', fontsize=30)
    plt.savefig('pbpb2760_dndeta.pdf')
    plt.show()

def cmp_ptspec(path_to_results='', cent = ['0-5', '5-10', '10-20']):
    from pbpb2760 import dNdPt
    exp = dNdPt()
    for i, c in enumerate(cent):
        if i == 0:
            label0 = r'$ALICE$'
            label1 = r'$CLVisc$'
        else:
            label0, label1 = None, None

        x, y, yerr0, yerr1 = exp.get('pion', c)
        plt.errorbar(x, y, yerr=(yerr0, yerr1), label=label0)
        path = os.path.join(path_to_results, c.replace('-', '_'))
        dndpt = ebe_mean(path, kind='dndpt', hadron='pion', rap='Y')
        print c, dndpt
        plt.semilogy(dndpt[:, 0], dndpt[:, 1], label=label1)
    plt.xlim(0, 3)
    plt.ylim(1.0E-2, 1.0E4)
    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$(1/2\pi)d^2 N_{\pi^+}/dYp_Tdp_T\ [GeV]^{-2}$')
    smash_style.set(line_styles=False)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV$', fontsize=30)
    plt.savefig('pbpb2760_ptspec.pdf')
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
    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV$', fontsize=30)
    plt.savefig('pbpb2760_pionv2.pdf')
    plt.show()




def ptspec_identify(path, cent='0_5', data_src=1):
    path = os.path.join(path, cent)
    pion = ebe_mean(path, kind='dndpt', hadron='pion', rap='Y')
    kaon = ebe_mean(path, kind='dndpt', hadron='kaon', rap='Y')
    proton = ebe_mean(path, kind='dndpt', hadron='proton', rap='Y')

    proton_fix_factor = 0.7
    if data_src == 0:
        #dat0 = np.loadtxt('dNdPt_2p76.dat', skiprows=10)
        dat = np.loadtxt( 'data/dNdYptdpt_Alice/dNdPt_pbpb2760_%s_pion_exp.dat'%cent, skiprows=10)
        dat2 = np.loadtxt('data/dNdYptdpt_Alice/dNdPt_pbpb2760_%s_kaon_exp.dat'%cent, skiprows=10)
        dat3 = np.loadtxt('data/dNdYptdpt_Alice/dNdPt_pbpb2760_%s_proton_exp.dat'%cent, skiprows=10)
        # 1304.0347
        #plt.errorbar(dat0[:,0], dat0[:,3], dat0[:,6], fmt='o', label=r'$ALICE\ charged$')
        plt.errorbar(dat[:,0], dat[:,3], dat[:,6], fmt='o',   color='r', label=r'$ALICE\ \pi^+$')
        plt.errorbar(dat2[:,0], dat2[:,3], dat2[:,6], fmt='s',color='g', label=r'$ALICE\ K^+$')
        plt.errorbar(dat3[:,0], dat3[:,3], dat3[:,6], fmt='d',color='b', label=r'$ALICE\ p$')
        #plt.semilogy(charged[:, 0], charged[:, 1], label=r'$CLVisc\ \pi^+$')
        plt.semilogy(pion[:, 0], pion[:, 1], 'k-', label=r'$CLVisc$')
        plt.semilogy(kaon[:, 0], kaon[:, 1], 'k-')
        plt.semilogy(proton[:, 0], proton_fix_factor*proton[:, 1], 'k-')
        #plt.semilogy(eta, 3.5*proton[:, 1], label=r'$CLVisc$')
    elif data_src == 1:
        from pbpb2760 import dNdPt
        exp = dNdPt()
        x_pion, y_pion, yerr0_pion, yerr1_pion = exp.get('pion', cent.replace('_', '-'))
        x_kaon, y_kaon, yerr0_kaon, yerr1_kaon = exp.get('kaon', cent.replace('_', '-'))
        x_proton, y_proton, yerr0_proton, yerr1_proton = exp.get('proton', cent.replace('_', '-'))
        plt.errorbar(x_pion, y_pion, yerr=(yerr0_pion, yerr1_pion), fmt='o',   color='r', label=r'$ALICE\ \pi^{+}+\pi^{-}$')
        plt.errorbar(x_kaon, y_kaon, yerr=(yerr0_kaon, yerr1_kaon), fmt='s',   color='g', label=r'$ALICE\ K^{+}+K^{-}$')
        plt.errorbar(x_proton, y_proton, yerr=(yerr0_proton, yerr1_proton), fmt='d',   color='b', label=r'$ALICE\ p+\bar{p}$')
        #plt.semilogy(charged[:, 0], charged[:, 1], label=r'$CLVisc\ \pi^+$')
        plt.semilogy(pion[:, 0], 2*pion[:, 1], 'k-', label=r'$CLVisc$')
        plt.semilogy(kaon[:, 0], 2*kaon[:, 1], 'k-')
        plt.semilogy(proton[:, 0],proton_fix_factor*2*proton[:, 1], 'k-')
        #plt.semilogy(eta, 3.5*proton[:, 1], label=r'$CLVisc$')

    plt.xlabel(r'$p_T\ [GeV]$')
    plt.ylabel(r'$\frac{dN}{2\pi dY p_Tdp_T}\ [GeV^{-2}]$')

    smash_style.set(line_styles=False)
    #plt.legend(loc='best', ncol=2, mode='expand')
    plt.title(r'$Pb+Pb\ 2.76\ TeV, centrality\ %s$'%cent.replace('_', '-'))
    plt.xlim(0, 4)
    plt.ylim(1.0E-2, 1.0E4)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig('pbpb2760_ptspec_%s_identify.pdf'%cent)
    plt.show()



if __name__=='__main__':
    path = "/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results/"
    #cmp_dndeta(path)
    #cmp_ptspec(path)
    ptspec_identify(path, cent='0_5')
    #ptspec_identify(path, cent='5_10')
    #cmp_v2_pion(path)
