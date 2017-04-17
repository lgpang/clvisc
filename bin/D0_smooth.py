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
from glauber import Glauber, weight_mean_b

def one_shot(fout, impact_parameter=7.8):
    if not os.path.exists(fout):
        os.makedirs(fout)
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 121

    cfg.DT = 0.005
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.DZ = 0.15
    cfg.IEOS = 1
    cfg.ntskip = 60
    cfg.nxskip = 4
    cfg.nyskip = 4
    cfg.nzskip = 2
    cfg.ImpactParameter = impact_parameter

    cfg.Hwn = 0.95

    cfg.TAU0 = 0.4
    cfg.Edmax = 55.0
    cfg.ETAOS = 0.20
    cfg.fPathOut = fout
    write_config(cfg)

    t0 = time()
    visc = CLVisc(cfg, gpu_id=2)
    ini = Glauber(cfg, visc.ctx, visc.queue, visc.compile_options,
                  visc.ideal.d_ev[1], save_nbc=True)
    #visc.create_ini_from_partons(fname_partons, SIGR=0.6, SIGZ=0.6, KFACTOR=1.2)
    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

    cwd = os.getcwd()
    os.chdir('../CLSmoothSpec/build')
    os.system('cmake -D VISCOUS_ON=ON ..')
    os.system('make')
    call(['./spec', fout])
    os.chdir(cwd)
    after_reso = '0'
    call(['python', '../spec/main.py', fout, after_reso])
    os.chdir(cwd)


if __name__ == '__main__':
    fpath_out = os.path.abspath('/lustre/nyx/hyihp/lpang/D0/etaos_0p20_check')

    cent_min = [0, 2, 4, 6, 8,  0,  10, 20, 30, 40, 50, 60, 70, 0, 40, 0]
    cent_max = [2, 4, 6, 8, 10, 10, 20, 30, 40, 50, 60, 70, 80, 40, 80, 80]
    #cent_min = [70]
    #cent_max = [80]

    for cmin, cmax in zip(cent_min, cent_max):
        b = weight_mean_b(cmin, cmax)
        fout = os.path.join(fpath_out, 'cent_%s_%s'%(cmin, cmax))
        one_shot(fout, b)
