#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 20 Nov 2015 12:04:35 CET
import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))

from ideal import CLIdeal
from config import cfg, write_config
import pyopencl as cl
from glauber import Glauber

def squeezing(tdec=0.1, eB0=0.1, sigx=1.3, sigy=2.6, path_out='../results/event0'):
    cfg.IEOS = 2
    cfg.TAUD = tdec
    cfg.EB0 = eB0
    cfg.SIGX = sigx
    cfg.SIGY = sigy
    cfg.fPathOut = path_out

    cfg.ImpactParameter = 10.0

    if not os.path.exists(cfg.fPathOut):
        os.mkdir(cfg.fPathOut)

    write_config(cfg)

    ideal = CLIdeal(cfg, gpu_id=2)
    
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines, ideal.d_ev[1])
    
    ideal.evolve(max_loops=2000, save_bulk=True, save_hypersf=True)
    
    #cl.enqueue_copy(ideal.queue, ideal.h_ev1, ideal.d_ev[1]).wait()


#### start the hydro with -DT
#cfg.TAU0 = ideal.
#cfg.DT = -cfg.DT
#cfg.fPathOut = '../results/event_reverse'
#inverse_time = CLIdeal(cfg)
#inverse_time.load_ini(ideal.h_ev1)
#
#inverse_time.evolve(max_loops=100, save_bulk=True, to_maxloop=True, save_hypersf=False)

squeezing(tdec=1.9, eB0=0.09, path_out='../results/squeezing_td1p9_eb0p09')
