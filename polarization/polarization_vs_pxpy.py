#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import matplotlib.pyplot as plt
import numpy as np
import math

from gpu_polarization import Polarization
import four_momentum as mom
from common_plotting import smash_style
from numba import jit
import matplotlib.pyplot as plt

import h5py

# store the data in hdf5 file
f_h5 = h5py.File('vor_vs_pxpy.hdf5', 'w')
N = 20
px = np.linspace(-3, 3, N)
py = np.linspace(-3, 3, N)
rapidity = np.linspace(-5, 5, 11, endpoint=True)

dset_px = f_h5.create_dataset('mom/px', data=px)
dset_py = f_h5.create_dataset('mom/py', data=py)
dset_rapidity = f_h5.create_dataset('mom/Y', data=rapidity)

def plot_vor(vor, Y, fpath='./'):
    vmax = vor.max()
    vmin = vor.min()
    if vmax < -vmin:
        vmax = -vmin
    plt.imshow(vor.T, cmap=plt.get_cmap('bwr'), origin='lower', extent=(-3,3,-3,3), vmin=-vmax, vmax=vmax)
    plt.xlabel(r'$p_x\ [GeV]$')
    plt.ylabel(r'$p_y\ [GeV]$')
    plt.title(r'$\Pi^{y}\ @\ rapidity=%s$'%Y)
    plt.colorbar()
    smash_style.set()
    plt.savefig('%s/vor_Y%s.png'%(fpath, Y))
    plt.close()



def polarization_vs_pxpy(fpath, event_id):
    '''calc the pt, phi integrated lambda polarization as a function of
    rapidity.
    The results is stored in hdf5 file'''
    sf = np.loadtxt('%s/hypersf.dat'%fpath, dtype=np.float32)
    omega = np.loadtxt('%s/omegamu_sf.dat'%fpath, dtype=np.float32)
    LambdaPolarization = Polarization(sf, omega)

    vor = np.zeros((N, N))
    rho = np.zeros((N, N))

    for Y in rapidity:
        for i, px_ in enumerate(px):
            for j, py_ in enumerate(py):
                pol_ij, omg_ij, rho_ij = LambdaPolarization.pol_vor_rho(Y, px_, py_)
                vor[i, j] = pol_ij
                rho[i, j] = rho_ij
        name = 'event%s/rapidity%s/vor_vs_px_py'%(event_id, Y)
        dset_vor = f_h5.create_dataset(name, data=vor)
        name = 'event%s/rapidity%s/rho_vs_px_py'%(event_id, Y)
        dset_rho = f_h5.create_dataset(name, data=rho)

        #plot_vor(vor/rho, Y)

        print(Y, 'finished')


if __name__ == '__main__':
    polarization_vs_pxpy(fpath='./', event_id=0)
    f_h5.close()

