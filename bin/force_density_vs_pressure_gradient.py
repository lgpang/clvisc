#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 06 Feb 2015 15:28:42 CET

import matplotlib.pyplot as plt
import numpy as np
import os, sys

hbarc = 0.1973
hbarc3 = hbarc*hbarc*hbarc


def eB(x, y, t, s_x=1.3, s_y=2.6, eB_0=1.33, t_d=1.9):
    ''' s_x: gaussian sigma in units of fm
        s_y: gaussian sigma in units of fm
        eB_0: = 5 mpi^2 ~ 0.9 GeV^2
        t_d: decay rate = 0.1...1 fm
        return (eB_x, eB_y) in units GeV^2'''
    return eB_0 * np.outer(np.exp(-x*x/(2*s_x*s_x)), np.exp(-y*y/(2*s_y*s_y))) * np.exp(-t/t_d)




def forcedensity(T, t):
    '''T: temperature in units of GeV
       t: time in units of fm
       return: 2 dimensional force density distribution in units GeV/fm^3'''
    chiT = 1/(3*np.pi*np.pi) * np.log(T/0.110)
    x = np.linspace(-16,16,401)
    y = np.linspace(-16,16,401)
    eBxy = eB(x,y,t)
    grad = np.gradient(eBxy, 0.08, 0.08)
    #return chiT*eBxy*np.sqrt(grad[0]*grad[0] + grad[1]*grad[1])/hbarc3
    return chiT*eBxy*grad[0]/hbarc3

def pressure_gradient(edxy):
    '''calc the pressure gradient with given transverse energy density'''
    cwd, cwf = os.path.split(__file__)
    sys.path.append(os.path.join(cwd, '..', 'pyvisc/'))
    from eos.eos import Eos
    ce = Eos(2)
    pressure = ce.f_P(edxy)
    grad = np.gradient(pressure, 0.08, 0.08)
    return grad[0], grad[1]


def plot_force_density():
    tau0 = 0.2
    T = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/Txy0.dat')
    F = forcedensity(T, tau0)
    extent = (-16, 16, -16, 16)
    plt.imshow(F.T, extent=extent, origin='lower')
    plt.colorbar()
    plt.title( r'force density in units GeV/fm^4 at $\tau$={tau:0.1f} fm'.format(tau=tau0) )
    plt.xlabel( 'x [fm]' )
    plt.ylabel( 'y [fm]' )
    plt.show()

def plot_pressure_gradient():
    tau0 = 0.2
    edxy = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/edxy0.dat')
    dpdx, dpdy = pressure_gradient(edxy)
    extent = (-16, 16, -16, 16)
    plt.imshow(-dpdx.T, extent=extent, origin='lower')
    plt.colorbar()
    plt.title( r'-dP/dx in units GeV/fm^4 at $\tau$={tau:0.1f} fm'.format(tau=tau0) )
    plt.xlabel( 'x [fm]' )
    plt.ylabel( 'y [fm]' )
    plt.show()

if __name__ == '__main__':
    #plot_pressure_gradient()
    plot_force_density()

