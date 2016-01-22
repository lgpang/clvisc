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
    # do some cutoff to temperature
    T[T<0.110+1.0E-6] = 0.110+1.0E-6

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


def plot_force_density(system='Pb+Pb', tau=0.2, s_x=1.3, s_y=2.6, eB0=1.33, lifetime=1.9):
    tau0 = tau
    T = None
    if system == 'Pb+Pb':
        T = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/Txy0.dat')
    else:
        T = np.loadtxt('../results/tau0_dependence/squeezing_auau200_tau0p2_td1p9_eb0p09/Txy0.dat')
    Fx, Fy = forcedensity(T, tau0, s_x, s_y, eB0, lifetime)
    x = np.linspace(-16, 16, 401)
    y = np.linspace(-16, 16, 401)
    x, y = np.meshgrid(x, y)

    #cs_dpdx = ax[0].contour(x, y, Fx.T)
    #ax[0].clabel(cs_dpdx, fontsize=16, inline=1, fmt='%1.1f')
    #cs_dpdy = ax[1].contour(x, y, Fy.T)
    #ax[1].clabel(cs_dpdy, fontsize=16, inline=1, fmt='%1.1f', levels=cs_dpdx.levels)

    #plt.figure(figsize=(8, 16))
    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    cs_dpdx = plt.contourf(x, y, Fx.T, cmap=plt.cm.bone, aspect='auto')
    cb = plt.colorbar(cs_dpdx)
    #cb.formatter.set_powerlimits((0, 0))
    #cb.update_ticks()

    plt.xlabel('$x\ [fm]$')
    plt.ylabel('$y\ [fm]$')
    rmax = 10
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)
    plt.text( -8, 7, r'(a) $F^x$', fontsize=45)
    plt.text( -9, -4, r'$%s$'%system)
    plt.text( -9, -6, r'$\tau=0.2$ fm')

    if system == 'Au+Au':
        plt.text( -9, -8, r'$settings\ (A)$')
    elif system == 'Pb+Pb':
        plt.text( -9, -8, r'$settings\ (E)$')

    font_size = 25
    cb.ax.tick_params(labelsize=font_size)

    plt.subplot(122)
    cs_dpdy = plt.contourf(x, y, Fy.T, cmap=plt.cm.bone, aspect='auto')

    cb = plt.colorbar(cs_dpdy)

    font_size = 25
    cb.ax.tick_params(labelsize=font_size)
    #cb.formatter.set_powerlimits((0, 0))
    #cb.update_ticks()

    plt.text( -8, 7, r'(b) $F^y$', fontsize=45)
    plt.xlabel('$x\ [fm]$')
    #plt.ylabel('$y\ [fm]$')
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)
    smash_style.set()
    #plt.subplots_adjust(left=0.20, right=0.93, top=0.95, bottom=0.1, hspace=0.3)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.15, hspace=0.3)
    plt.show()


def plot_pressure_gradient(system='Au+Au'):
    tau0 = 0.2
    edxy = None
    if system == 'Pb+Pb':
        edxy = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/edxy0.dat')
    else:
        edxy = np.loadtxt('../results/tau0_dependence/squeezing_auau200_tau0p2_td1p9_eb0p09/edxy0.dat')
    dpdx, dpdy = pressure_gradient(edxy)
    extent = (-16, 16, -16, 16)
    plt.figure(figsize=(20,8))
    plt.subplot(121)

    x = np.linspace(-16, 16, 401)
    y = np.linspace(-16, 16, 401)
    x, y = np.meshgrid(x, y)

    rmax = 10
    cs_dpdx = plt.contourf(x, y, -dpdx.T, cmap=plt.cm.bone, aspect='auto')
    cb = plt.colorbar(cs_dpdx)
    plt.text( -9, 8, r'(a) $-\partial_xP$', fontsize=45)
    plt.text( -9, -4, r'$%s$'%system)
    plt.text( -9, -6, r'$\tau=0.2$ fm')
    plt.text( -9, -8, r'$b=10$ fm')
    plt.xlabel('$x\ [fm]$')
    plt.ylabel('$y\ [fm]$')
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)

    font_size = 25
    cb.ax.tick_params(labelsize=font_size)

    plt.subplot(122)
    cs_dpdy = plt.contourf(x, y, -dpdy.T, cmap=plt.cm.bone, aspect='auto')
    cb = plt.colorbar(cs_dpdy)
    plt.text( -9, 8, r'(b) $-\partial_yP$', fontsize=45)
    plt.xlabel('$x\ [fm]$')
    #plt.ylabel('$y\ [fm]$')
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)

    cb.ax.tick_params(labelsize=font_size)

    smash_style.set()

    plt.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.15, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    #plot_pressure_gradient('Au+Au')
    plot_pressure_gradient('Pb+Pb')
    #plot_force_density(system='Au+Au', eB0=0.09)
    #plot_force_density(system='Pb+Pb', eB0=1.33)

