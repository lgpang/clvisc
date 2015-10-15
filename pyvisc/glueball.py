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
    cfg.Edmax = eos.f_ed(Tmax)

    ideal = CLIdeal(cfg)
    from glauber import Glauber
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines,
                  ideal.d_ev[1])

    ideal.evolve(max_loops=1000)

    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0 ))



if __name__=='__main__':
    glueball(0.6, '../results/AATmax0p6')


