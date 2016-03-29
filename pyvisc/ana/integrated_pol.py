#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import matplotlib.pyplot as plt
import numpy as np
import math

from polarization import Polarization
import four_momentum as mom
from common_plotting import smash_style
from numba import jit
import matplotlib.pyplot as plt
import os
import h5py


def init_momentum(f_h5, rapidity):
    dset_pt = f_h5.create_dataset('mom/PT', data=mom.PT)
    dset_phi = f_h5.create_dataset('mom/PHI', data=mom.PHI)
    dset_rapidity = f_h5.create_dataset('mom/Y', data=rapidity)

def integrated_polarization(f_h5, fpath, event_id, rapidity):
    '''calc the pt, phi integrated lambda polarization as a function of
    rapidity. The results is stored in hdf5 file'''
    sf = np.loadtxt('%s/hypersf.dat'%fpath, dtype=np.float32)
    omega = np.loadtxt('%s/omegamu_sf.dat'%fpath, dtype=np.float32)
    LambdaPolarization = Polarization(sf, omega)

    npt, nphi = mom.NPT, mom.NPHI


    vor = np.zeros((npt, nphi))
    rho = np.zeros((npt, nphi))
    vor_int, rho_int = [], []

    nrap = len(rapidity)
    momentum_list = np.zeros((nrap*npt*nphi, 4), dtype=np.float32)
    mass = 1.115

    for k, Y in enumerate(rapidity):
        for i, pt in enumerate(mom.PT):
            for j, phi in enumerate(mom.PHI):
                px = pt * math.cos(phi)
                py = pt * math.sin(phi)
                mt = math.sqrt(mass*mass + px*px + py*py)
                index = k*npt*nphi + i*nphi + j

                momentum_list[index, 0] = mt
                momentum_list[index, 1] = Y
                momentum_list[index, 2] = px
                momentum_list[index, 3] = py

    pol, rho, pol_lrf = LambdaPolarization.get(momentum_list)

    #name = 'event%s/rapidity%s/vor_vs_pt_phi'%(event_id, Y)
    #dset_vor = f_h5.create_dataset(name, data=vor)
    #name = 'event%s/rapidity%s/rho_vs_pt_phi'%(event_id, Y)
    #dset_rho = f_h5.create_dataset(name, data=rho)

    piy = pol_lrf[:, 2].reshape(nrap, npt, nphi)
    for k, Y in enumerate(rapidity):
        #vor_int.append( mom.pt_phi_integral(vor) )
        #rho_int.append( mom.pt_phi_integral(rho) )
        vor_int.append(mom.pt_phi_integral(piy[k, :, :]))
        print(Y, 'finished')

    name = 'event%s/integral_pt_phi/vor'%event_id
    dset_vorint = f_h5.create_dataset(name, data=vor_int)
    #name = 'event%s/integral_pt_phi/rho'%event_id
    #dset_rhoint = f_h5.create_dataset(name, data=rho_int)
    print('event', event_id, 'finished')



def create_file(fname, rapidity):
    f_h5 = h5py.File(fname, 'w')
    init_momentum(f_h5, rapidity)
    return f_h5

def update_h5(start_id, end_id, f_h5name, path, rapidity, create=False):
    # store the data in hdf5 file
    f_h5 = None
    if not os.path.exists(f_h5name) or create:
        f_h5 = create_file(f_h5name, rapidity)
    else:
        f_h5 = h5py.File(f_h5name, 'r+')

    for event_id in range(start_id, end_id):
        fpath = '%s/event%s'%(path, event_id)
        integrated_polarization(f_h5, fpath, event_id, rapidity)
        print('event', event_id, 'finished')
    f_h5.close()



def fin_grid_mid_rapidity():
    rapidity = np.linspace(-5, 5, 40, endpoint=True)
    f_h5name = 'vor_int_visc0p08_auau200_cent20_30.hdf5'
    path = '/lustre/nyx/hyihp/lpang/auau200_results/cent20_30/etas0p08/'
    update_h5(0, 100, f_h5name, path, rapidity, create=True)


fin_grid_mid_rapidity()
