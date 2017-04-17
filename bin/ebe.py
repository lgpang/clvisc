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



def read_p4x4(cent='30_35', idx=0,
        fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    '''read 4-momentum and 4-coordiantes from h5 file,
    return: np.array with shape (num_of_partons, 8)
    the first 4 columns store: E, px, py, pz
    the last 4 columns store: t, x, y, z'''
    with h5py.File(fname, 'r') as f:
        grp = f['cent']
        event_id = grp[cent][:, 0].astype(np.int)

        impact = grp[cent][:, 1]
        nw = grp[cent][:, 2]
        nparton = grp[cent][:, 3]
        key = 'event%s'%event_id[idx]
        #print key, nw[0], nparton[0]
        p4x4 = f[key]
        return p4x4[...], event_id[idx], impact[idx], nw[idx], nparton[idx]

def event_by_event(fout, cent='30_35', idx=0, etaos=0.0,
                   fname_ini='/lustre/nyx/hyihp/lpang/hdf5_data/auau39.h5',
                   gpu_id = 3):
    ''' Run event_by_event hydro, with initial condition 
    from smearing on the particle list'''
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 181

    cfg.DT = 0.005
    cfg.DX = 0.1
    cfg.DY = 0.1
    cfg.DZ = 0.1
    cfg.IEOS = 1
    cfg.TFRZ = 0.137

    cfg.ntskip = 60
    cfg.nzskip = 3

    cfg.TAU0 = 0.4
    cfg.ETAOS = etaos
    cfg.fPathOut = fout

    t0 = time()
    visc = CLVisc(cfg, gpu_id=gpu_id)

    parton_list, eid, imp_b, nwound, npartons = read_p4x4(cent, idx, fname_ini)

    comments = 'cent=%s, eventid=%s, impact parameter=%s, nw=%s, npartons=%s'%(
            cent, eid, imp_b, nwound, npartons)

    write_config(cfg, comments)

    visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.5)

    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=True)

    # test whether queue.finish() fix the opencl memory leak problem
    visc.queue.finish()

    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 7:
        print("Usage: python ebe.py collision_system centrality_range  etaos gpuid start_id end_id")
        exit()

    collision_system = sys.argv[1]
    cent = sys.argv[2]
    etaos = np.float32(sys.argv[3])
    gpuid = int(sys.argv[4])
    start_id = int(sys.argv[5])
    end_id = int(sys.argv[6])

    path = '/lustre/nyx/hyihp/lpang/%s_results/cent%s/etas%s/'%(collision_system,
              cent, etaos)
    path = path.replace('.', 'p')

    fname_ini='/lustre/nyx/hyihp/lpang/hdf5_data/%s.h5'%collision_system

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("path exists, may be created just now by another thread")

    for idx in xrange(start_id, end_id):
        fpath_out = path + 'event%s'%(idx)
        event_by_event(fpath_out, cent, idx, etaos=etaos,
                       fname_ini=fname_ini, gpu_id=gpuid)

        cwd = os.getcwd()
        os.chdir('../CLSmoothSpec/build')
        #os.system('cmake -D VISCOUS_ON=ON ..')
        #os.system('make')
        call(['./spec', fpath_out])
        os.chdir(cwd)
        call(['python', '../spec/main.py', fpath_out])
