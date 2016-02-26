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

cfg.IEOS = 0

cfg.NX = 401
cfg.NY = 401
cfg.NZ = 1

cfg.TAU0 = 0.6
cfg.Edmax = 17.0
cfg.Hwn = 1.0
cfg.ImpactParameter = 7.0

write_config(cfg)

ideal = CLIdeal(cfg)

from glauber import Glauber
ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines, ideal.d_ev[1])

ideal.evolve(max_loops=1000, save_bulk=True, to_maxloop=True, save_hypersf=False)


