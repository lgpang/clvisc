#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Di 01 Dez 2015 13:52:39 CET

import matplotlib.pyplot as plt
import numpy as np

from common_plotting import smash_style


#Tx = np.loadtxt('CompareWithHarri_SU3_T0p6/Tx.dat')
#Tx = np.loadtxt('../results/SU3_T0p6/Tx.dat')
#Tx = np.loadtxt('../results/Glueball_T0p6/Tx.dat')
#Tx = np.loadtxt('../results/Lattice2p1_T0p6/Tx.dat')
Tx = np.loadtxt('../results/Lattice2p1_Edmax166/Tx.dat')
#Tx = np.loadtxt('event0/Tx.dat')
#Tx = np.loadtxt('../results/PP_SU3/Tx.dat')

extent = (-19, 19, 0.4, 30)

CI = plt.imshow(Tx.T, origin='lower', extent=extent, aspect='auto', vmax=0.449, vmin=0.001)

#levels = np.array([0.40, 0.30, 0.271, 0.26, 0.24, 0.15])
#levels = np.array([0.27, 0.25, 0.15])
#levels = np.array([0.275, 0.220, 0.150])
levels = np.array([0.220, 0.150])

#CS = plt.contour(Tx.T, levels, origin='lower', lw=8, extent=extent, colors=('k', 'g', 'r'), vmin=0.1, vmax=0.55)
CS = plt.contour(Tx.T, levels, origin='lower', lw=8, extent=extent, colors=('g', 'r'))

plt.clabel(CS, levels, inline=1, fmt='%0.3f', fontsize=20)

plt.title(r'$(b)\ 2+1\ Flavor\ QCD$')
#plt.title(r'$(a)\ Pure\ SU3\ gauge\ EOS$')


#plt.text(0, 28, r'$\tau_0=0.4\ fm,\ T_0=0.6\ GeV$')


#plt.xlim(0, 20)

#smash_style.set(minorticks_on=False)

if __name__ == '__main__':
    cbar = plt.colorbar(CI)
    plt.subplots_adjust(bottom=0.15)
    
    plt.xlabel(r'$r\ [fm]$')
    plt.ylabel(r'$\tau$ [fm]')
    plt.text(21, 31, r'$T\ [GeV]$', fontsize=35)
    #plt.xlim(-7, 7)
    #plt.ylim(0.6, 7)
    
    smash_style.set()
    #plt.savefig('pp_YM.pdf')
    #plt.savefig('PP_QCD_LG.pdf')
    
    #plt.savefig('QCD_LG_T300MeV.pdf')
    plt.savefig('QCD_LG_Ed166.pdf')
    
    plt.show()
    
