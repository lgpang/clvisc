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
from backend_opencl import backend

class TestBjorken(unittest.TestCase):
    def setUp(self):
        bend = backend(cfg, gpu_id=0)
        self.ideal = CLIdeal(configs=cfg, backend=bend)
        self.ctx = bend.ctx
        self.queue = bend.default_queue


    def test_bjorken(self):
        ''' initialize with uniform energy density in (tau, x, y, eta) coordinates
        to test the Bjorken expansion:
           eps/eps0 = (tau/tau0)**(-4.0/3.0)
        '''

        with open('../kernel/kernel_hyperfs.cl', 'r') as f:
            kernel_src = f.read()

        cwd, cwf = os.path.split(__file__)
    
        compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
        compile_options.append('-D USE_SINGLE_PRECISION')
        compile_options.append('-D nx_skip=10')
        compile_options.append('-D ny_skip=10')
        compile_options.append('-D nz_skip=10')
        compile_options.append('-D EFRZ=2.5f')
        compile_options.append('-D EOSI')
        prg = cl.Program(self.ctx, kernel_src).build(compile_options)

        final = np.empty(32).astype( array.vec.float4 )
        mf = cl.mem_flags
        final_gpu = cl.Buffer(self.ctx, mf.READ_WRITE, final.nbytes)
 
        prg.test_hypersf(self.queue, (1,), None, final_gpu)
        cl.enqueue_copy(self.queue, final, final_gpu).wait()
        print('the 3d volume of one cube is:')
        print(final[0])

   

if __name__ == '__main__':
    unittest.main()
