#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob
import pyopencl as cl
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp2d

import os, sys
cwd, cwf = os.path.split(__file__)
print('cwd=', cwd)

sys.path.append(os.path.join(cwd, '../pyvisc'))

from eos.eos import Eos

pce = Eos(1)

def qgp_fraction(T):
    '''calc the QGP fraction from temperature'''
    frac = np.zeros_like(T)
    
    frac[T>0.22] = 1.0
    frac[T<0.165] = 0.0

    cross_over = np.logical_and(T>=0.165, T<=0.22)
    frac[cross_over] = (T[cross_over] - 0.165)/(0.22 - 0.165)

    return frac



def interp_2d(dat_input, x_input, y_input,
              xmin=-9.75, xmax=9.75, nx=66, ymin=-9.75, ymax=9.75, ny=66):
    x = np.linspace(xmin, xmax, nx, endpoint=True)
    y = np.linspace(ymin, ymax, ny, endpoint=True)
    f = interp2d(x_input, y_input, dat_input)
    dat_output = f(x, y)
    return x, y, dat_output


def create_table_for_jet(fpath):
    import cStringIO
    output = cStringIO.StringIO()

    fname = os.path.join(fpath, 'bulkinfo.h5')
    with h5py.File(fname, 'r') as h5:
        tau_list = h5['coord/tau'][...]
        x_list = h5['coord/x'][...]
        y_list = h5['coord/y'][...]
        for tau in tau_list:
            tau_str = ('%s'%tau).replace('.', 'p')
            ed = h5['bulk2d/exy_tau%s'%tau_str][...]
            vx = h5['bulk2d/vx_xy_tau%s'%tau_str][...]
            vy = h5['bulk2d/vy_xy_tau%s'%tau_str][...]
            T  = pce.f_T(ed)
            QGP_fraction = qgp_fraction(T)

            x, y, ed_new = interp_2d(ed, x_list, y_list)
            x, y, vx_new = interp_2d(vx, x_list, y_list)
            x, y, vy_new = interp_2d(vy, x_list, y_list)
            x, y, T_new = interp_2d(T, x_list, y_list)
            x, y, frac_new = interp_2d(QGP_fraction, x_list, y_list)

            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    print >> output, tau, xi, yj, ed_new[i, j], T_new[i, j], vx_new[i, j], vy_new[i, j], frac_new[i, j], 0.0

        with open(os.path.join(fpath, 'bulk.dat'), 'w') as f:
            f.write(output.getvalue())



if __name__ == '__main__':
    import h5py

    h5 = h5py.File('/lustre/nyx/hyihp/lpang/pbpb5p02/pbpb2p76/pbpb2p76.h5', 'r+')

    cents = ['%s_%s'%(i*10, (i+1)*10) for i in range(9)]

    harmonic_orders = [2, 3]

    for cent in cents:
        print('start ', cent)
        for n in harmonic_orders:
            for idx in range(19):
                fpath = os.path.join('/lustre/nyx/hyihp/lpang/pbpb5p02/pbpb2p76',
                        cent, 'n%s'%n, 'bin%s'%idx)

                create_table_for_jet(fpath)

                bulk_path = cent + '/ecc%s/bulk_bin%s'%(n, idx)
                h5.create_dataset(bulk_path, data=np.loadtxt(os.path.join(fpath, 'bulk.dat')))

                nch_path = cent + '/ecc%s/spec_bin%s/dndeta'%(n, idx)
                h5.create_dataset(nch_path, data=np.loadtxt(os.path.join(fpath, 'dNdEta_mc_charged.dat')))

                spec_path = cent + '/ecc%s/spec_bin%s/ptspec/'%(n, idx)

                names = ['charge', 'pion_plus', 'kaon_plus', 'proton']
                files = ['dN_over_2pidEtaptdpt_mc_charged.dat', 'dN_over_2pidYptdpt_mc_211.dat',
                         'dN_over_2pidYptdpt_mc_321.dat', 'dN_over_2pidYptdpt_mc_2212.dat']

                for mm in range(4):
                    h5.create_dataset(spec_path + names[mm], data=np.loadtxt(os.path.join(fpath, files[mm])))

                print('idx=', idx)


