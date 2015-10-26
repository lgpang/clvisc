#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import pyopencl as cl
from pyopencl import array
import os, sys
from time import time
import numpy as np
import unittest

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from ideal import CLIdeal
from config import cfg

class TestBjorken(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.cfg.NX = 5
        self.cfg.NY = 5
        self.cfg.NZ = 5
        self.cfg.IEOS = 0
        self.ideal = CLIdeal(self.cfg)
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue


    def test_bjorken(self):
        ''' initialize with uniform energy density in (tau, x, y, eta) coordinates
        to test the Bjorken expansion:
           eps/eps0 = (tau/tau0)**(-4.0/3.0)
        '''

        kernel_src = """
        # include "real_type.h"
        __kernel void init_ev(global real4 * d_ev1,
                   const int size) {
          int gid = (int) get_global_id(0);
          if ( gid < size ) {
             d_ev1[gid] = (real4)(30.0f, 0.0f, 0.0f, 0.0f);
          }
        }
        """
        cwd, cwf = os.path.split(__file__)
    
        compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
        compile_options.append('-D USE_SINGLE_PRECISION')
        prg = cl.Program(self.ctx, kernel_src).build(compile_options)
        prg.init_ev(self.queue, (self.ideal.size,), None, self.ideal.d_ev[1],
                    np.int32(self.ideal.size)).wait()

        self.ideal.evolve(max_loops=200, save_bulk=False,
                          save_hypersf=False)

        history = np.array(self.ideal.history)
        tau, edmax = history[:,0], history[:,1]
        a = (tau/tau[0])**(-4.0/3.0)
        b = edmax/edmax[0]
        np.testing.assert_almost_equal(a, b, 3)
    

if __name__ == '__main__':
    unittest.main()
