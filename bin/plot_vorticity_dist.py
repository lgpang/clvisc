#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Di 12 Jan 2016 13:43:30 CET

import matplotlib.pyplot as plt
import numpy as np



'''comments:
    I can do integration over the freeze out hyper surface with
    weight $p.dSigma f(p.u) omega_xz$ to calculate the azimuthal
    and rapidity dependence of vorticity'''

def plot_vorticity_on_sf():
    #sf = np.loadtxt('../results/event1_ideal_noiniflow/hypersf.dat')
    #vorticity = np.loadtxt('../results/event1_ideal_noiniflow/vorticity_xz.dat')
    sf = np.loadtxt('../results/event1_ideal/hypersf.dat')
    vorticity = np.loadtxt('../results/event1_ideal/vorticity_xz.dat')
    
    vx = sf[:, 4]
    vy = sf[:, 5]
    vetas = sf[:, 6]
    etas = sf[:, 7]
    
    phi = np.arctan2(vy, vx)
    Y = np.arctanh(vetas) + etas

    sel = np.sqrt(vx*vx+vy*vy) > 0.1

    plt.hist2d(phi[sel], Y[sel], bins=20, weights=vorticity[sel], vmin=-150, vmax=150)
    plt.colorbar()
    #plt.hist(Y, bins=20, weights=vorticity)
    #plt.hist(phi, bins=20, weights=vorticity)
    plt.xlabel(r'$\phi$', fontsize=25)
    plt.ylabel(r'$Y$', fontsize=25)
    plt.title(r'$\omega_{xz}$ on freezeout hypersf', fontsize=25)
    plt.show()

plot_vorticity_on_sf()
