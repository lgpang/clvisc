'''Test the eos from image2d_t table'''

import os, sys

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from eos.eos import Eos

eos = Eos(eos_type="pure_gauge")

eos.test_eos(3.0)

print('EFRZ(Tfrz=0.137)=', eos.f_ed(0.137))
