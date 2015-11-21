#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob

import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg
from visc import CLVisc

def event_by_event(fname_partons, fout):
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 81

    cfg.DT = 0.005
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.IEOS = 2
    cfg.ntskip = 60

    cfg.TAU0 = 0.4
    cfg.ETAOS = 0.16
    cfg.fPathOut = fout

    t0 = time()
    visc = CLVisc(cfg)
    visc.create_ini_from_partons(fname_partons, SIGR=0.6, SIGZ=0.6, KFACTOR=1.2)
    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=False)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

finis = glob('/u/lpang/AuAu200_0_80/P*.txt')

for i, fname in enumerate(finis):
    fname_partons = fname
    fpath_out = '/u/lpang/PyVisc/results/D0/event%d/'%i
    
    event_by_event(fname_partons, fpath_out)
    
    cwd = os.getcwd()
    
    os.chdir('../CLSmoothSpec/build')
    #os.system('cmake -D VISCOUS_ON=ON ..')
    #os.system('make')
    call(['./spec', fpath_out])
    os.chdir(cwd)
    call(['python', '../spec/main.py', fpath_out])
