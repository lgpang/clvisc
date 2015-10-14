'''Test the eos from image2d_t table'''

import os, sys
import pyopencl as cl

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from config import cfg
from eos.eos import Eos

compile_options = []


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print cfg.IEOS

cfg.IEOS = 2

eos = Eos(cfg, ctx, queue, compile_options)

eos.test_eos(cfg, 3.0)

print('EFRZ(Tfrz=0.137)=', eos.efrz(0.137))
