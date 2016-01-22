#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 17 Dez 2015 16:17:07 CET

import matplotlib.pyplot as plt
import numpy as np
import pylab

import force_density_vs_pressure_gradient as squeezing

def subplot(x, y, text, sigx=1.3, sigy=2.6):
    eB0 = 1.33
    tau0 = 0.2
    s_x, s_y = sigx, sigy
    lifetime = 1.9
    T = np.loadtxt('../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33/Txy0.dat')
    Fx, Fy = squeezing.forcedensity(T, tau0, s_x, s_y, eB0, lifetime)

    cs_dpdx = plt.contourf(x, y, Fx.T, cmap=plt.cm.bone, aspect='auto')
    cb = plt.colorbar(cs_dpdx)
    rmax = 10
    plt.xlim(-rmax, rmax)
    plt.ylim(-rmax, rmax)
    plt.text(-9, 7, r'%s'%text)

    font_size = 15
    cb.ax.tick_params(labelsize=font_size)



def plot_force_density_vs_td():
    '''plot fx vs td for Pb+Pb collisions'''

    x = np.linspace(-16, 16, 401)
    y = np.linspace(-16, 16, 401)
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 8))

    plt.subplot(221)
    subplot(x, y, text=r'(F) $t_d=0.1\ fm$', sigx=1.3, sigy=2.6)
    plt.ylabel('$y\ [fm]$')

    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.subplot(222)
    subplot(x, y, text='(D) $t_d=0.5\ fm$', sigx=1.5, sigy=3.0)

    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.subplot(223)
    subplot(x, y, text='(E) $t_d=1.0\ fm$', sigx=1.8, sigy=3.6)

    plt.xlabel('$x\ [fm]$')
    plt.ylabel('$y\ [fm]$')

    plt.subplot(224)
    subplot(x, y, text='(F) $t_d=1.9\ fm$', sigx=2.4, sigy=4.8)
    plt.xlabel('$x\ [fm]$')
    plt.setp(plt.gca().get_yticklabels(), visible=False)

    plt.suptitle(r'$f^x\ for\ Pb+Pb\ collisions\ at\ \tau_0=0.2\ fm$')

    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.90, wspace=0.2, hspace=0.2)

    plt.show()


if __name__ == '__main__':
    plot_force_density_vs_td()
