#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Di 01 Dez 2015 13:52:39 CET

import matplotlib.pyplot as plt
import numpy as np

from common_plotting import smash_style

plot_cs2 = False

Tx = np.loadtxt('../results/Glueball_Edmax166/Tx.dat')

#Tx = np.loadtxt('../results/Lattice2p1_T0p6/Tx.dat')

#Tx = np.loadtxt('event0/Tx.dat')
#Tx = np.loadtxt('../results/PP_SU3/Tx.dat')

extent = (-19, 19, 0.4, 30)


#levels = np.array([0.40, 0.30, 0.271, 0.26, 0.24, 0.15])
#levels = np.array([0.27, 0.25, 0.15])

#CS = plt.contour(Tx.T, levels, origin='lower', lw=8, extent=extent, colors=('k', 'g', 'r'), vmin=0.1, vmax=0.55)

from scipy.interpolate import interp1d
eos_su3 = np.loadtxt('glueball_v2.dat')

T = eos_su3[:, 4] * 1.0E-3
cs2 = eos_su3[:, 5]

f_cs2 = interp1d(T, cs2)

if plot_cs2:
    levels = np.array([0.07])
    CI = plt.imshow(f_cs2(Tx.T), origin='lower', extent=extent, aspect='auto', vmax=0.149, vmin=0.0)
    CS = plt.contour(f_cs2(Tx.T), levels, origin='lower', lw=8, extent=extent, colors=('k', 'g', 'r'))
    plt.clabel(CS, levels, inline=1, fmt='%0.2f', fontsize=20)
    plt.title(r'$\ Pure\ SU3\ gauge\ EOS$')
else:
    CI = plt.imshow(Tx.T, origin='lower', extent=extent, aspect='auto', vmax=0.449, vmin=0.001)
    levels = np.array([0.275, 0.26, 0.15])
    CS = plt.contour(Tx.T, levels, origin='lower', lw=8, extent=extent, colors=('k', 'g', 'r'))
    plt.clabel(CS, levels, inline=1, fmt='%0.3f', fontsize=20)
    plt.title(r'$(a)\ Pure\ SU3\ Gauge$')




if __name__ == '__main__':
    cbar = plt.colorbar(CI)
    plt.xlabel(r'$r\ [fm]$')
    plt.ylabel(r'$\tau$ [fm]')

    if plot_cs2:
        plt.text(23, 31, r'$c_s^2$', fontsize=35)
    else:
        plt.text(21, 31, r'$T\ [GeV]$', fontsize=35)
   
    plt.subplots_adjust(bottom=0.15)
    
    smash_style.set()
    #plt.xlim(-7, 7)
    #plt.ylim(0.6, 7)
    
    #plt.savefig('pp_YM.pdf')
    #plt.savefig('PP_QCD_LG.pdf')
    
    #plt.savefig('SU3_LG_T300MeV.pdf')
    if plot_cs2:
        #plt.savefig('SU3_LG_T300MeV_CS2.pdf')
        plt.savefig('SU3_LG_Ed166_CS2.pdf')
    else:
        #plt.savefig('SU3_LG_T300MeV.pdf')
        plt.savefig('SU3_LG_Ed166.pdf')
    
    
    #plt.savefig('QCD_LG.pdf')
    
    plt.show()
    
