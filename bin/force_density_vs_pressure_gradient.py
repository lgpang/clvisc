#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 06 Feb 2015 15:28:42 CET

from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from common_plotting import smash_style

hbarc = 0.1973
hbarc3 = hbarc*hbarc*hbarc


def eB(x, y, t, s_x=1.3, s_y=2.6, eB_0=1.33, t_d=1.9):
    ''' s_x: gaussian sigma in units of fm
        s_y: gaussian sigma in units of fm
        eB_0: = 5 mpi^2 ~ 0.9 GeV^2
        t_d: decay rate = 0.1...1 fm
        return (eB_x, eB_y) in units GeV^2'''
    return eB_0 * np.outer(np.exp(-x*x/(2*s_x*s_x)), np.exp(-y*y/(2*s_y*s_y))) * np.exp(-t/t_d)




def forcedensity(T, t, s_x, s_y, eB_0, t_d):
    '''T: temperature in units of GeV
       t: time in units of fm
       return: 2 dimensional force density distribution in units GeV/fm^3'''
    chiT = 1/(3*np.pi*np.pi) * np.log(T/0.110)
    x = np.linspace(-16,16,401)
    y = np.linspace(-16,16,401)
    eBxy = eB(x,y,t, s_x, s_y, eB_0, t_d)
    grad = np.gradient(eBxy, 0.08, 0.08)
    #return chiT*eBxy*np.sqrt(grad[0]*grad[0] + grad[1]*grad[1])/hbarc3
    return chiT*eBxy*grad[0]/hbarc3, chiT*eBxy*grad[1]/hbarc3

def pressure_gradient(edxy):
    '''calc the pressure gradient with given transverse energy density'''
    cwd, cwf = os.path.split(__file__)
    sys.path.append(os.path.join(cwd, '..', 'pyvisc/'))
    from eos.eos import Eos
    ce = Eos(2)
    pressure = ce.f_P(edxy)
    grad = np.gradient(pressure, 0.08, 0.08)
    return grad[0], grad[1]


def plot_force_density(tau=0.2, system='Pb+Pb', s_x=1.3, s_y=2.6, eB0=1.33, lifetime=1.9):
    tau0 = tau
    #T = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/Txy0.dat')
    T = np.loadtxt('../results/tau0_dependence/squeezing_auau200_tau0p2_td1p9_eb0p09/Txy0.dat')
    Fx, Fy = forcedensity(T, tau0, s_x, s_y, eB0, lifetime)
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].text( -4.5, 4, r'(a) $f^x$')
    ax[0].text( -4.5, -2, r'$%s$'%system)
    ax[0].text( -4.5, -3, r'$\tau=0.2$ fm')
    ax[0].text( -4.5, -4, r'$settings\ A$')

    x = np.linspace(-16, 16, 401)
    y = np.linspace(-16, 16, 401)
    x, y = np.meshgrid(x, y)
    cs_dpdx = ax[0].contour(x, y, Fx.T, colors='k', linewidths=3)
    ax[0].clabel(cs_dpdx, fontsize=16, inline=1, fmt='%1.1f')

    cs_dpdy = ax[1].contour(x, y, Fy.T, colors='k', linewidths=3)
    ax[1].clabel(cs_dpdy, fontsize=16, inline=1, fmt='%1.1f')


    #im1 = ax[1].imshow(-dpdy.T, extent=extent,cmap=colormap, origin='lower')
    ax[1].text( -4.5, 4, r'(b) $f^y$')
    ax[1].text( -4.5, -2, r'$%s$'%system)
    ax[1].text( -4.5, -3, r'$\tau=0.2$ fm')
    ax[1].text( -4.5, -4, r'$settings\ A$')

    ax[0].set_xlabel('$x\ [fm]$')
    ax[0].set_ylabel('$y\ [fm]$')
    ax[1].set_xlabel('$x\ [fm]$')
    ax[1].set_ylabel('$y\ [fm]$')

    ax[0].set_xlim(-5.2, 5.2)
    ax[1].set_xlim(-5.2, 5.2)
    ax[0].set_ylim(-5.2, 5.2)
    ax[1].set_ylim(-5.2, 5.2)
    #smash_style.set()
    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.15, wspace=0.4)
    plt.show()


def plot_pressure_gradient(system='Au+Au'):
    tau0 = 0.2
    #edxy = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/edxy0.dat')
    edxy = np.loadtxt('../results/tau0_dependence/squeezing_auau200_tau0p2_td1p9_eb0p09/edxy0.dat')
    dpdx, dpdy = pressure_gradient(edxy)
    extent = (-16, 16, -16, 16)

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    #ax[0].imshow(-dpdx.T, extent=extent, cmap=colormap, origin='lower')

    ax[0].text( -9, 8, r'(a) $-\partial_xP$')
    ax[0].text( -9, -4, r'$%s$'%system)
    ax[0].text( -9, -6, r'$\tau=0.2$ fm')
    ax[0].text( -9, -8, r'$b=10$ fm')

    x = np.linspace(-16, 16, 401)
    y = np.linspace(-16, 16, 401)
    x, y = np.meshgrid(x, y)
    cs_dpdx = ax[0].contour(x, y, -dpdx.T, colors='k', linewidths=3)
    ax[0].clabel(cs_dpdx, fontsize=16, inline=1, fmt='%1.0f')

    cs_dpdy = ax[1].contour(x, y, -dpdy.T, colors='k', linewidths=3)
    ax[1].clabel(cs_dpdy, fontsize=16, inline=1, fmt='%1.0f')


    #im1 = ax[1].imshow(-dpdy.T, extent=extent,cmap=colormap, origin='lower')
    ax[1].text( -9, 8, r'(b) $-\partial_yP\ [GeV/fm^4]$')
    ax[1].text( -9, -4, r'$%s$'%system)
    ax[1].text( -9, -6, r'$\tau=0.2$ fm')
    ax[1].text( -9, -8, r'$b=10$ fm')

    ax[0].set_xlabel('$x\ [fm]$')
    ax[0].set_ylabel('$y\ [fm]$')
    ax[1].set_xlabel('$x\ [fm]$')
    ax[1].set_ylabel('$y\ [fm]$')

    ax[0].set_xlim(-10., 10.)
    ax[1].set_xlim(-10., 10.)
    ax[0].set_ylim(-10., 10.)
    ax[1].set_ylim(-10., 10.)
    smash_style.set()
    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.15, wspace=0.4)
    plt.show()

if __name__ == '__main__':
    #plot_pressure_gradient('Au+Au')
    plot_force_density(system='Au+Au')

