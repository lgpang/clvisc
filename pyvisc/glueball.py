#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 16 Oct 2015 12:07:35 AM CEST

from ideal import CLIdeal
from eos.eos import Eos
from time import time
import os
from subprocess import call
from config import cfg, write_config

def glueball(Tmax = 0.6, outdir = '../results/event0'):
    print('start ...')
    t0 = time()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cfg.IEOS = 3
    eos = Eos(cfg.IEOS)
    # update the configuration
    cfg.Edmax = eos.f_ed(Tmax)
    cfg.fPathOut = outdir

    # set IEOS = 2 for (2+1)-flavor QCD EOS
    # set IEOS = 3 for GlueBall EOS
    cfg.NX = 501
    cfg.NY = 501
    cfg.NZ = 1
    cfg.DT = 0.01
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.A = 208
    cfg.Ra = 6.62
    cfg.Eta = 0.546
    cfg.Si0 = 6.4
    cfg.TAU0 = 0.2
    cfg.ImpactParameter = 7.74
    cfg.ETAOS = 0.0
    cfg.SQRTS = 2760

    #cfg.Edmax = 600.0
    cfg.Hwn = 1.0
    write_config(cfg)

    ideal = CLIdeal(cfg, gpu_id=2)
    from glauber import Glauber
    Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines,
                  ideal.d_ev[1])

    ideal.evolve(max_loops=3000, save_hypersf=False, to_maxloop=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0 ))



if __name__=='__main__':
    #glueball(0.60, '../results/IdealGas_Ed30/')
    #glueball(0.50, '../results/IdealGas_T0p5/')
    #glueball(0.40, '../results/IdealGas_T0p4/')
    #glueball(0.30, '../results/IdealGas_T0p3/')
    glueball(0.60, '../results/CompareWithHarri_SU3_T0p6/')
