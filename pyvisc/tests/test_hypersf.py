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

class TestHypersf(unittest.TestCase):
    def setUp(self):
        self.ideal = CLIdeal(configs=cfg)
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue


    def test_hypersf(self):
        ''' initialize 4D cube with ed=2 at (n=0, i,j,k=*),
        ed=3 at (n=1, i,j,k=*) and EFRZ=2.5 to calc the volumn of
        3D cube at ed=2.5 freeze out hypersurface in 4D space.
        '''
        with open('../kernel/kernel_hypersf.cl', 'r') as f:
            kernel_src = f.read()

        cwd, cwf = os.path.split(__file__)
    
        compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
        compile_options.append('-D USE_SINGLE_PRECISION')
        compile_options.append('-D NX=205')
        compile_options.append('-D NY=205')
        compile_options.append('-D NZ=85')
        compile_options.append('-D nxskip=10')
        compile_options.append('-D nyskip=10')
        compile_options.append('-D nzskip=10')
        compile_options.append('-D DT=0.02')
        compile_options.append('-D DX=0.16')
        compile_options.append('-D DY=0.16')
        compile_options.append('-D DZ=0.3')
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
