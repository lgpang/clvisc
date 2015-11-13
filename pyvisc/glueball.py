#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 16 Oct 2015 12:07:35 AM CEST

from ideal import CLIdeal
from eos.eos import Eos
from time import time
import os
from subprocess import call

def glueball(Tmax = 0.6, outdir = '../results/event0'):
    from config import cfg
    print('start ...')
    t0 = time()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cfg.IEOS = 2
    eos = Eos(cfg.IEOS)
    # update the configuration
    cfg.Edmax = eos.f_ed(Tmax)
    cfg.fPathOut = outdir

    # set IEOS = 2 for (2+1)-flavor QCD EOS
    # set IEOS = 3 for GlueBall EOS
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 81
    cfg.DT = 0.02
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.ImpactParameter = 3.0

    ideal = CLIdeal(cfg)
    from glauber import Glauber
    Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines,
                  ideal.d_ev[1])

    ideal.evolve(max_loops=1000, save_hypersf=False, to_maxloop=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0 ))



if __name__=='__main__':
    path = '/scratch/hyihp/pang/ini/GlueBall/'
    glueball(0.60, path+'QCD_AATmax0p6_b7fm')
    #glueball(0.45, path+'AATmax0p45_b7fm')
    #glueball(0.3, path+'AATmax0p3_b7fm')
    #glueball(0.55, path+'AATmax0p55')
    #glueball(0.50, path+'AATmax0p5')
    #glueball(0.45, path+'AATmax0p45')
    #glueball(0.40, path+'AATmax0p4')
    #glueball(0.35, path+'AATmax0p35')
    #glueball(0.30, path+'AATmax0p3')
    #glueball(0.250,  path+'QCD_AATmax0p25')
