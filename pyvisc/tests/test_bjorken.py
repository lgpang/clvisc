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
from backend_opencl import OpenCLBackend

class TestBjorken(unittest.TestCase):
    def setUp(self):
        cfg.NX = 25
        cfg.NY = 25
        cfg.NZ = 25
        cfg.BSZ = 32
        cfg.IEOS = 0
        cfg.opencl_interactive = True
        backend = OpenCLBackend(cfg, gpu_id=0)
        self.ideal = CLIdeal(cfg, backend)
        self.ctx = backend.ctx
        self.queue = backend.default_queue


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

        self.ideal.evolve(max_loops=200)
        history = np.array(self.ideal.history)
        tau, edmax = history[:,0], history[:,1]
        print(edmax)
        a = (tau/tau[0])**(-4.0/3.0)
        b = edmax/edmax[0]
        np.testing.assert_almost_equal(a, b, 2)
    

if __name__ == '__main__':
    unittest.main()
