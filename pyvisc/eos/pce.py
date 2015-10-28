#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 10 Dec 2014 10:04:43 AM CST

import matplotlib.pyplot as plt
import numpy as np

class lattice_qcd(object):
    '''chemical eqlibrium and non-chemical equilibrium EOS from lattice QCD
    Table is made by Pasi Huovinien'''
    def __init__(self, kind='CE'):
        if kind=='CE':
            eps0 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_dens1.dat', skiprows=2)
            eps1 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_dens2.dat', skiprows=2)
            eps2 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_dens3.dat', skiprows=2)
            eps3 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_dens4.dat', skiprows=2)
            T0 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_par1.dat', skiprows=2)
            T1 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_par2.dat', skiprows=2)
            T2 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_par3.dat', skiprows=2)
            T3 = np.loadtxt( 'eos_table/s95p-v1/s95p-v1_par4.dat', skiprows=2)
        elif kind=='PCE':
            eps0 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_dens1.dat', skiprows=2)
            eps1 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_dens2.dat', skiprows=2)
            eps2 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_dens3.dat', skiprows=2)
            eps3 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_dens4.dat', skiprows=2)
            T0 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_par1.dat', skiprows=2)
            T1 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_par2.dat', skiprows=2)
            T2 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_par3.dat', skiprows=2)
            T3 = np.loadtxt( 'eos_table/s95p-PCE165-v0/s95p-PCE165-v0_par4.dat', skiprows=2)

        self.eps_dat = [eps0, eps1, eps2, eps3]
        self.T_dat = [T0, T1, T2, T3]

    def plot(self):
        for i in range(4):
            eps = self.eps_dat[i]
            print('len of eos table ', len(eps[:,0]))
            plt.plot(eps[:,0], eps[:,1])
        plt.show()


CE = lattice_qcd('CE')
PCE = lattice_qcd('PCE')
CE.plot()
PCE.plot()
