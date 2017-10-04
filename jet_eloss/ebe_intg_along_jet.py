#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 01 Sep 2017 05:24:10 AM CEST

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d, splrep, splint
from scipy.integrate import quad
from four_momentum import PT, PHI, NY, NPT, NPHI, HBARC


def integrate_along_jet(spec_along_phi, positive_x=True, quick_integral=False, jet_direction=0.0):
    '''Args:
          spec_along_phi: 1D spectra with 48 components along phi from [0, 2pi]
       Return:
          The integrated spectra for phi in [-pi/2, pi/2] '''
    angles = np.concatenate([-PHI[::-1][:-1], PHI, PHI+2*np.pi])
    specs = np.concatenate([spec_along_phi[::-1][:-1], spec_along_phi, spec_along_phi])
    #tck = splrep(PHI, spec_along_phi, per=True)
    if quick_integral:
        tck = splrep(angles, specs)
        if positive_x:
            return splint(jet_direction - 0.5*np.pi, jet_direction + 0.5*np.pi, tck) / np.pi
        else:
            return splint(jet_direction + 0.5*np.pi, jet_direction + 1.5*np.pi, tck) / np.pi
    else:
        fn = interp1d(angles, specs)
        if positive_x:
            return quad(fn, jet_direction - 0.5*np.pi, jet_direction + 0.5*np.pi, epsrel=1.0E-6)[0] / np.pi
        else:
            return quad(fn, jet_direction + 0.5*np.pi, jet_direction + 1.5*np.pi, epsrel=1.0E-6)[0] / np.pi


def ptspec_along_jet(path, jet_direction, positive_x=True, quick_integral=False):
    '''positive_x=True for along jet;
       positive_x=False for opposite to jet;'''
    full_spec = np.loadtxt(os.path.join(path, 'dNdYPtdPtdPhi_211.dat')).reshape(NY, NPT, NPHI)[20, :, :]/(HBARC)**3.0
    ptspec = np.apply_along_axis(arr=full_spec, func1d=integrate_along_jet,
                               axis=1, positive_x=positive_x, quick_integral=quick_integral,
                               jet_direction=jet_direction)
    if positive_x:
        np.savetxt(os.path.join(path, 'piplus_spec_along_jet.txt'), np.array([PT, ptspec]).T)
    else:
        np.savetxt(os.path.join(path, 'piplus_spec_opposite_to_jet.txt'), np.array([PT, ptspec]).T)
    

def create_spec_along_jet(fpaths, jet_directions):
    for idx, path in enumerate(fpaths):
        jet_direction = jet_directions[idx]
        ptspec_along_jet(path, jet_direction, positive_x=True, quick_integral=False)
        ptspec_along_jet(path, jet_direction, positive_x=False, quick_integral=False)
        print(idx, 'finished')


if __name__ == '__main__':
    fpaths_ideal = ['/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.0_with/event%s'%i for i in range(100)]
    fpaths_visc = ['/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.16_with/event%s'%i for i in range(100)]
    fpaths = fpaths_ideal + fpaths_visc
    jet_100_directions = np.loadtxt('random_angles.txt')[:100]
    jet_directions = np.concatenate([jet_100_directions, jet_100_directions, np.zeros(2)])
    print(jet_directions)
    fpaths.append('/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.0/')
    fpaths.append('/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.16/')
    create_spec_along_jet(fpaths, jet_directions)

    #collect_ptspec(fin_name='piplus_spec_along_jet.txt', fout_name='pt_spec_diff_along_jet.txt')


