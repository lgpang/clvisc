import pyopencl as cl
from pyopencl import array
import numpy as np
from time import time
import os

import unittest

class TestReductionMethod(unittest.TestCase):
  def test_reduction(self):
    os.environ['PYOPENCL_CTX'] = ':1'
    ctx = cl.create_some_context()
    queue = cl.CommandQueue( ctx )
    mf = cl.mem_flags
    N = 50000
    xr = np.random.rand(N).astype( np.float32 )
    t1 = time()
    xmax = xr.max()
    t2 = time()
    x = np.empty(N, array.vec.float4)
    for i in range(N):
        x[i] = (xr[i], 0.0, 0.0, 0.0)

    x_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    cwd, cwf = os.path.split(__file__)

    kernel_src = open(os.path.join(cwd, '..', 'kernel', 'kernel_reduction.cl'), 'r').read()

    compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
    compile_options.append('-D USE_SINGLE_PRECISION' )

    prg = cl.Program(ctx, kernel_src ).build(compile_options)

    num_of_groups = 64
    work_group_size = 64

    y_semi = np.zeros(num_of_groups).astype( np.float32 )
    y_gpu = cl.Buffer( ctx, mf.READ_WRITE, y_semi.nbytes )
    
    globalsize = num_of_groups * work_group_size
    
    final = np.empty(1).astype( np.float32 )
    final_gpu = cl.Buffer( ctx, mf.READ_WRITE, final.nbytes )
    
    prg.reduction_stage1( queue, (globalsize,), (64,), x_gpu, y_gpu, np.int32(N) )
    
    prg.reduction_stage2( queue, (64,), (64,), y_gpu, final_gpu )
    
    cl.enqueue_read_buffer( queue, final_gpu, final ).wait()
    t3 = time()
    self.assertAlmostEqual(xmax, final)
    print("reduction test passed")

if __name__ == '__main__':
    unittest.main()
