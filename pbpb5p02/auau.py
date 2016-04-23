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

import os, sys
cwd, cwf = os.path.split(__file__)
print('cwd=', cwd)

sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc


def auau_collisions(fout, cent_min=0, cent_max=5, edmax=85,
                    idx=0, etaos=0.0, gpu_id = 3):
    ''' Run event_by_event hydro, with initial condition 
    from smearing on the particle list'''
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 361
    cfg.NY = 361
    cfg.NZ = 121

    cfg.DT = 0.01
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.DZ = 0.15
    cfg.IEOS = 1
    cfg.TFRZ = 0.137

    cfg.A = 197
    cfg.Ra = 6.38
    cfg.Edmax = edmax
    cfg.Eta = 0.535
    cfg.Si0 = 4.0
    cfg.Eta_gw = 1.0
    cfg.Eta_flat = 1.5

    cfg.ntskip = 30
    cfg.nxskip = 4
    cfg.nyskip = 4
    cfg.nzskip = 2

    cfg.Hwn = 0.95

    cfg.TAU0 = 0.6
    cfg.ETAOS = etaos
    cfg.fPathOut = fout

    cfg.save_to_hdf5 = True

    comments = 'au+au test'

    t0 = time()

    visc = CLVisc(cfg, gpu_id=gpu_id)

    visc.optical_glauber_ini(system='Au+Au', cent_min=cent_min, cent_max=cent_max,
                             save_binary_collisions=True)


    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=False)

    t1 = time()

    write_config(cfg, comments)

    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

def main(edmax=98, idx=0, cent_min=0, cent_max=5, gpuid=0, etaos=0.08):
    path = '/lustre/nyx/hyihp/lpang/pbpb5p02'
    if not os.path.exists(path):
        os.makedirs(path)

    fpath_out = os.path.join(os.path.abspath(path), 'event%s/'%(idx))
    auau_collisions(fpath_out, cent_min, cent_max, edmax, idx, etaos=etaos,
                       gpu_id=gpuid)
    from create_table import create_table_for_jet
    create_table_for_jet(fpath_out)

    cwd = os.getcwd()
    #os.chdir('../CLSmoothSpec/build')
    #os.system('cmake -D VISCOUS_ON=ON ..')
    #os.system('make')
    #call(['./spec', fpath_out])
    #os.chdir(cwd)
    #after_reso = '0'
    #call(['python', '../spec/main.py', fpath_out, after_reso])

    os.chdir('../sampler/plot/')
    viscous_on = 'true'
    after_reso = 'true'
    call(['python', 'dNdY_test.py', fpath_out, viscous_on, after_reso])

    os.chdir(cwd)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 6:
        print('usage: python pbpb.py edmax idx cent_min cent_max gpuid')

    edmax = float(sys.argv[1])
    idx = int(sys.argv[2])
    cent_min = int(sys.argv[3])
    cent_max = int(sys.argv[4])
    gpuid = int(sys.argv[5])

    main(edmax, idx,  cent_min, cent_max, gpuid)



