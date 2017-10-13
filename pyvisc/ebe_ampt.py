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

sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc



def create_longitudinal_profile(cfg):
    ''' create longitudinal_profile according to cfg.Eta_flat, cfg.Eta_gw,
    cfg.NZ and cfg.DZ '''
    eta_max = cfg.NZ//2 * cfg.DZ
    eta = np.linspace(-eta_max, eta_max, cfg.NZ)

    heta = np.ones(cfg.NZ)

    fall_off = np.abs(eta) > cfg.Eta_flat
    eta_fall = np.abs(eta[fall_off])
    heta[fall_off] = np.exp(-(eta_fall - cfg.Eta_flat)**2/(2.0*cfg.Eta_gw**2))
    return heta



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
        p4x4 = f[key]
        return p4x4[...], event_id[idx], impact[idx], nw[idx], nparton[idx]

def event_by_event(fout, cent='30_35', idx=0, etaos=0.0, system = 'auau200',
                   fname_ini='/lustre/nyx/hyihp/lpang/hdf5_data/auau39.h5', gpu_id = 3,
                   switch_off_longitudinal_fluctuations = False, force_bjorken = False, IEOS=1):
    ''' Run event_by_event hydro, with initial condition from smearing on the particle list'''
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 121

    cfg.DT = 0.01
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.DZ = 0.16
    cfg.ntskip = 32
    cfg.nzskip = 2
    cfg.nxskip = 2
    cfg.nyskip = 2

    #cfg.NX = 301
    #cfg.NY = 301
    #cfg.NZ = 51

    #cfg.DT = 0.005
    #cfg.DX = 0.1
    #cfg.DY = 0.1
    #cfg.DZ = 0.2

    ## IEOS = 1 for default
    #cfg.IEOS = 1
    cfg.IEOS = IEOS

    #cfg.TFRZ = 0.110
    #cfg.TFRZ = 0.105

    #cfg.TFRZ = 0.137
    cfg.TFRZ = 0.100

    cfg.TAU0 = 0.4


    if system == 'pbpb2p76':
        cfg.TAU0 = 0.2
        cfg.Eta_gw = 1.8
        cfg.Eta_flat = 2.0
    elif system == 'auau200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5

    #cfg.ETAOS = etaos

    cfg.ETAOS_XMIN = 0.154
    cfg.ETAOS_YMIN = etaos
    cfg.ETAOS_RIGHT_SLOP = 0.0
    cfg.ETAOS_LEFT_SLOP =  0.0


    cfg.fPathOut = fout

    t0 = time()
    visc = CLVisc(cfg, gpu_id=gpu_id)

    parton_list, eid, imp_b, nwound, npartons = read_p4x4(cent, idx, fname_ini)

    comments = 'cent=%s, eventid=%s, impact parameter=%s, nw=%s, npartons=%s'%(
            cent, eid, imp_b, nwound, npartons)

    write_config(cfg, comments)

    if force_bjorken:
        if cfg.IEOS == 1:
            # KFACTOR=1.4 for etaos = 0.08; KFACTOR = 1.2 for etaos=0.16
            visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.2, force_bjorken=True)
        elif cfg.IEOS == 5:
            visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=0.8, force_bjorken=True)
    elif switch_off_longitudinal_fluctuations:
        heta = create_longitudinal_profile(cfg)
        visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.4, longitudinal_profile=heta)
    else:
        if etaos >= 0.16:
            visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.2)
        elif etaos >= 0.08:
            visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.4)
        else:
            visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.5)

    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=True)

    # test whether queue.finish() fix the opencl memory leak problem
    visc.queue.finish()

    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 8:
        print("Usage: python ebe.py collision_system centrality_range  etaos")
        exit()

    collision_system = sys.argv[1]
    cent = sys.argv[2]
    etaos = np.float32(sys.argv[3])
    gpuid = int(sys.argv[4])
    start_id = int(sys.argv[5])
    end_id = int(sys.argv[6])
    IEOS = int(sys.argv[7])

    path = '/lustre/nyx/hyihp/lpang/trento_ebe_hydro/%s_results_ampt/etas%s/%s/'%(collision_system,
              etaos, cent)
    path = path.replace('.', 'p')

    fname_ini='/lustre/nyx/hyihp/lpang/hdf5_data/%s.h5'%collision_system

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print("path exists, may be created just now by another thread")

    for idx in xrange(start_id, end_id):
        fpath_out = path + 'event%s'%(idx)
        # skip the events from a previous run
        if os.path.exists(fpath_out):
            continue

        try:
            event_by_event(fpath_out, cent, idx, etaos=etaos, system=collision_system,
                       fname_ini=fname_ini, gpu_id=gpuid, IEOS=IEOS)
        except:
                print("Unexpected error:", sys.exc_info()[0])

        viscous_on = "true"
        if etaos < 0.0001: viscous_on = "false"
        # get particle spectra from MC sampling and force decay
        call(['python', 'spec.py', '--event_dir', cfg.fPathOut,
          '--viscous_on', viscous_on, "--reso_decay", "true", "--nsampling", "2000",
          '--mode', 'mc'])
    
         # calc the smooth particle spectra
        call(['python', 'spec.py', '--event_dir', cfg.fPathOut,
          '--viscous_on', viscous_on, "--reso_decay", "false", 
          '--mode', 'smooth'])
