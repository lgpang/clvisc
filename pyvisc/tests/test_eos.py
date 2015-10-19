'''Test the eos from image2d_t table'''

import os, sys
import pyopencl as cl

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from config import cfg
from eos.eos import Eos

cfg.IEOS = 3

eos = Eos(cfg)

eos.test_eos(3.0)

print('EFRZ(Tfrz=0.137)=', eos.f_ed(0.137))
