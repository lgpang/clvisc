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
from ini.trento import AuAu200, PbPb2760, PbPb5020
from scipy.interpolate import InterpolatedUnivariateSpline

import os, sys
cwd, cwf = os.path.split(__file__)
print('cwd=', cwd)

sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc

import h5py

def from_sd_to_ed(entropy, eos):
    '''using eos to  convert the entropy density to energy density'''
    s = eos.s
    ed = eos.ed
    # the InterpolatedUnivariateSpline works for both interpolation
    # and extrapolation
    f_ed = InterpolatedUnivariateSpline(s, ed, k=1)
    return f_ed(entropy)


def ebehydro(fpath, cent='0_5', etaos=0.12, gpu_id=0, system='pbpb2760', oneshot=False):
    ''' Run event_by_event hydro, with initial condition 
    from smearing on the particle list'''

    fout = fpath
    if not os.path.exists(fout):
        os.mkdir(fout)

    cfg.NX = 200
    cfg.NY = 200
    cfg.NZ = 121
    cfg.DT = 0.02
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.DZ = 0.20

    cfg.ntskip = 20
    cfg.nxskip = 2
    cfg.nyskip = 2
    cfg.nzskip = 2

    cfg.IEOS = 1
    cfg.TAU0 = 0.6
    cfg.fPathOut = fout

    #cfg.TFRZ = 0.137
    cfg.TFRZ = 0.100

    cfg.ETAOS_XMIN = 0.154

    cfg.ETAOS_YMIN = 0.15
    cfg.ETAOS_RIGHT_SLOP = 0.0
    cfg.ETAOS_LEFT_SLOP =  0.0

    cfg.save_to_hdf5 = True

    # for auau
    if system == 'auau200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'au+au IP-Glasma'
        collision = AuAu200()
        scale_factor = 57.0
    # for pbpb
    else:
        cfg.Eta_gw = 1.8
        cfg.Eta_flat = 2.0
        comments = 'pb+pb IP-Glasma'
        if system == 'pbpb2760':
            collision = PbPb2760()
            scale_factor = 128.0
        elif system == 'pbpb5020':
            collision = PbPb5020()
            scale_factor = 151.0

    grid_max = cfg.NX/2 * cfg.DX

    fini = os.path.join(fout, 'trento_ini/')

    if os.path.exists(fini):
        call(['rm', '-r', fini])

    collision.create_ini(cent, fini, num_of_events=1,
                         grid_max=grid_max, grid_step=cfg.DX,
                         one_shot_ini=oneshot)
    if oneshot:
        s = np.loadtxt(os.path.join(fini, 'one_shot_ini.dat'))
    else:
        s = np.loadtxt(os.path.join(fini, '0.dat'))
    smax = s.max()
    s_scale = s * scale_factor
    t0 = time()

    visc = CLVisc(cfg, gpu_id=gpu_id)

    ed = from_sd_to_ed(s_scale, visc.ideal.eos)

    ev = np.zeros((cfg.NX*cfg.NY*cfg.NZ, 4), cfg.real)

    # repeat the ed(x,y) NZ times
    ev[:, 0] = np.repeat((ed.T).flatten(), cfg.NZ)

    eta_max = cfg.NZ//2 * cfg.DZ
    eta = np.linspace(-eta_max, eta_max, cfg.NZ)

    heta = np.ones(cfg.NZ)

    fall_off = np.abs(eta) > cfg.Eta_flat
    eta_fall = np.abs(eta[fall_off])
    heta[fall_off] = np.exp(-(eta_fall - cfg.Eta_flat)**2/(2.0*cfg.Eta_gw**2))

    # apply the heta longitudinal distribution
    ev[:, 0] *= np.tile(heta, cfg.NX * cfg.NY)

    visc.ideal.load_ini(ev)

    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=False)

    write_config(cfg, comments)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

    cwd = os.getcwd()
    os.chdir('../sampler/mcspec/')
    viscous_on = 'true'
    after_reso = 'true'
    nsampling = '2000'
    call(['python', 'sampler.py', fout, viscous_on, after_reso, nsampling])
    os.chdir(cwd)

    #from create_table import create_table_for_jet
    #create_table_for_jet(fout, visc.ideal.eos)
    os.chdir('../CLSmoothSpec/build')
    #os.system('cmake -D VISCOUS_ON=ON ..')
    #os.system('make')
    call(['./spec', fpath])
    os.chdir(cwd)
    after_reso = '0'
    call(['python', '../spec/main.py', fpath, after_reso])

def main(cent='0_5', gpu_id=0, jobs_per_gpu=25, system='pbpb2760'):
    path = '/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results/%s'%cent
    fpath_out = os.path.abspath(path)
    for i in xrange(gpu_id*jobs_per_gpu, (gpu_id+1)*jobs_per_gpu):
        fout = os.path.join(fpath_out, 'event%s'%i)
        if not os.path.exists(fout):
            os.makedirs(fout)
        ebehydro(fout, cent, system = system)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 4:
        cent = sys.argv[1]
        gpu_id = int(sys.argv[2])
        jobs_per_gpu = int(sys.argv[3])
        main(cent, gpu_id=gpu_id, jobs_per_gpu=jobs_per_gpu, system='pbpb5020')
