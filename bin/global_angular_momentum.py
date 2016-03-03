#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob
import pyopencl as cl
import matplotlib.pyplot as plt

import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc

def event_by_event(fname_partons, fout, etaos=0.0):
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 101

    cfg.DT = 0.005
    cfg.DX = 0.1
    cfg.DY = 0.1
    cfg.DZ = 0.15
    cfg.IEOS = 4
    cfg.TFRZ = 0.136

    cfg.ntskip = 60
    cfg.nzskip = 2

    cfg.TAU0 = 0.4
    cfg.ETAOS = etaos
    cfg.fPathOut = fout

    write_config(cfg)

    t0 = time()
    visc = CLVisc(cfg, gpu_id=3)
    visc.create_ini_from_partons(fname_partons, SIGR=0.6, SIGZ=0.6, KFACTOR=1.3)
    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))


def get_orbital_angular_momentum(fname_partons):
    '''calculate the initial orbital angular momentum
    for minimum bias Au+Au 200 GeV collisions
    return:
        jy = -tau0 * \int dx dy deta x * sinh(eta) * ed(x,y,eta)
        in units of hbar; where GeV*fm = 5 hbar'''
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 61

    cfg.DT = 0.005
    cfg.DX = 0.1
    cfg.DY = 0.1
    cfg.IEOS = 0
    cfg.ntskip = 60

    cfg.TAU0 = 0.4
    cfg.ETAOS = 0.0

    write_config(cfg)

    t0 = time()
    visc = CLVisc(cfg, gpu_id=1)
    visc.create_ini_from_partons(fname_partons, SIGR=0.6, SIGZ=0.6, KFACTOR=1.3)
    visc.ideal.ev_to_host()
    ed = visc.ideal.h_ev1[:,0].reshape(cfg.NX, cfg.NY, cfg.NZ)
    x = np.linspace(-15, 15, cfg.NX, endpoint=True)
    y = np.ones(cfg.NY)
    eta_s = np.linspace(-9, 9, cfg.NZ, endpoint=True)

    xx, yy, hh = np.meshgrid(x, y, eta_s, indexing='ij')

    jy = - xx * np.sinh(hh) * ed * cfg.TAU0 * cfg.DX * cfg.DY * cfg.DZ

    #plt.imshow(xx[:,cfg.NY/2,:], origin='lower')
    #plt.colorbar()
    #plt.show()
    return jy.sum()*5.0



def one_hydro_event():    
    fname = '/u/lpang/AuAu200_0_80/P30.txt'
    fpath_out = '../results/P30_WB_etaos0p08'
    event_by_event(fname, fpath_out)


def get_num_of_partons(fname):
    nparton = 0
    with open(fname, 'r') as fin:
        nparton = int(fin.readline())
        print nparton
    return nparton

def get_jy_vs_nparton():
    '''get orbital angular momentum as a function of number of partons'''
    finis = glob('/u/lpang/AuAu200_0_80/P*.txt')
    nparton, jy = [], []
    for i, fname in enumerate(finis):
        jy.append(get_orbital_angular_momentum(fname))
        nparton.append(get_num_of_partons(fname))
    np.savetxt('jy_vs_np.dat', np.array(zip(nparton, jy)),
               fmt='%.3f', header='nparton, obital angular momentum')

one_hydro_event()

#get_jy_vs_nparton()

#fname = '/u/lpang/AuAu200_0_80/P30.txt'
#get_orbital_angular_momentum(fname)

#finis = glob('/u/lpang/AuAu200_0_80/P*.txt')
#for i, fname in enumerate(finis[0:1]):
#    print fname
#    fname_partons = fname
#    fpath_out = '/tmp/lgpang/vorticity/event%d/'%i
#    
#    event_by_event(fname_partons, fpath_out)
#    
#    cwd = os.getcwd()
#    
#    #os.chdir('../CLSmoothSpec/build')
#    #os.system('cmake -D VISCOUS_ON=ON ..')
#    #os.system('make')
#    #call(['./spec', fpath_out])
#    #os.chdir(cwd)
#    #call(['python', '../spec/main.py', fpath_out])
#    if i > 50:
#        break
