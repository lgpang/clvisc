#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 20 Nov 2015 12:04:35 CET
import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))

from ideal import CLIdeal
from config import cfg
import pyopencl as cl

cfg.IEOS = 0

cfg.NX = 1
cfg.NY = 1
cfg.NZ = 1

ideal = CLIdeal(cfg)

from glauber import Glauber
ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines, ideal.d_ev[1])

ideal.evolve(max_loops=100, save_bulk=True, to_maxloop=True, save_hypersf=False)

cl.enqueue_copy(ideal.queue, ideal.h_ev1, ideal.d_ev[1]).wait()


#### start the hydro with -DT
cfg.TAU0 = ideal.tau
cfg.DT = -cfg.DT
cfg.fPathOut = '../results/event_reverse'
inverse_time = CLIdeal(cfg)
inverse_time.load_ini(ideal.h_ev1)

inverse_time.evolve(max_loops=100, save_bulk=True, to_maxloop=True, save_hypersf=False)
