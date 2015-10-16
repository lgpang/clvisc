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

    eos = Eos(cfg)
    # update the configuration
    cfg.Edmax = eos.f_ed(Tmax)
    cfg.fPathOut = outdir

    cfg.IEOS = 2

    ideal = CLIdeal(cfg)
    from glauber import Glauber
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines,
                  ideal.d_ev[1])

    ideal.evolve(max_loops=2000)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0 ))



if __name__=='__main__':
    path = '/scratch/hyihp/pang/ini/GlueBall/'
    glueball(0.55, path+'QCD_AATmax0p55')
    glueball(0.5,  path+'QCD_AATmax0p5')
    glueball(0.45, path+'QCD_AATmax0p45')
    glueball(0.4,  path+'QCD_AATmax0p4')
    glueball(0.35, path+'QCD_AATmax0p35')
    glueball(0.3,  path+'QCD_AATmax0p3')
    glueball(0.250,  path+'QCD_AATmax0p25')
