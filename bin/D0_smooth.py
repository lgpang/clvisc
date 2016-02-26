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
from config import cfg, write_config
from visc import CLVisc

def one_shot(fout):
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 61

    cfg.DT = 0.005
    cfg.DX = 0.10
    cfg.DY = 0.10
    cfg.IEOS = 1
    cfg.ntskip = 60
    cfg.nxskip = 3
    cfg.nyskip = 3
    cfg.ImpactParameter = 7.8

    cfg.TAU0 = 0.4
    cfg.Edmax = 55.0
    cfg.ETAOS = 0.18
    cfg.fPathOut = fout
    write_config(cfg)

    t0 = time()
    visc = CLVisc(cfg, gpu_id=1)
    from glauber import Glauber
    ini = Glauber(cfg, visc.ctx, visc.queue, visc.compile_options,
                  visc.ideal.d_ev[1])
    #visc.create_ini_from_partons(fname_partons, SIGR=0.6, SIGZ=0.6, KFACTOR=1.2)
    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

if __name__ == '__main__':
    fpath_out = '/tmp/lgpang/D0'
    one_shot(fpath_out)
    
    cwd = os.getcwd()
    
    os.chdir('../CLSmoothSpec/build')
    call(['./spec', fpath_out])

    os.chdir(cwd)
    call(['python', '../spec/main.py', fpath_out])
