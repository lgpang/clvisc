import pyopencl as cl
from pyopencl import array
import numpy as np
from time import time
import os, sys

sys.path.append('..')
from config import cfg
from eos.eos import Eos
import unittest

class TestHelper(unittest.TestCase):
  def setUp(self):
    os.environ['PYOPENCL_CTX'] = ':1'
    os.environ['PYOPENCL_COMPILER_OUTPUT']='1'

    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)

  def test_minmod(self):
    cwd, cwf = os.path.split(__file__)

    kernel_src = """
    #include "helper.h"
    
    __kernel void minmod_test(global real4 * result) {
      int gid = (int) get_global_id(0);
      if ( gid == 0 ) {
           real4 a = (real4)(1.0f, -2.0f, -1.0f, 4.0f);
           real4 b = (real4)(2.0f, 0.0f, -2.0f, 4.0f);
           real4 c = (real4)(3.0f, 2.0f, -3.0f, 4.0f);
           real4 d = minmod4(minmod4(a, b), c);
           result[0] = d;
      }
    }
    """
    compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
    compile_options.append('-D USE_SINGLE_PRECISION')
    compile_options.append('-D EOSI')

    prg = cl.Program(self.ctx, kernel_src ).build(compile_options)
   
    final = np.empty(1).astype( array.vec.float4 )
    mf = cl.mem_flags
    final_gpu = cl.Buffer(self.ctx, mf.READ_WRITE, final.nbytes)
    
    prg.minmod_test(self.queue, (1,), None, final_gpu)
    
    cl.enqueue_copy(self.queue, final, final_gpu).wait()

    self.assertAlmostEqual(final[0][0], 1.0)
    self.assertAlmostEqual(final[0][1], 0.0)
    self.assertAlmostEqual(final[0][2], -1.0)
    self.assertAlmostEqual(final[0][3], 4.0)
    print('minmod4 test pass')

  def test_rootfinding(self):
    cwd, cwf = os.path.split(__file__)

    kernel_src = """
    #include "helper.h"
    
    __kernel void rootfinding_test(
             global real4 * d_edv,
             global real * result,
             const int size,
             read_only image2d_t eos_table) {
      int gid = (int) get_global_id(0);
      if ( gid < size ) {
           real4 edv = d_edv[gid];
           real eps = edv.s0;
           real pre = P(eps, eos_table);
           real4 umu = (real4)(1.0f, edv.s1, edv.s2, edv.s3);
           real u0 = gamma(umu.s1, umu.s2, umu.s3);
           umu = u0*umu;
           real4 T0m = (eps+pre)*u0*umu - pre*gm[0];
           real M = sqrt(T0m.s1*T0m.s1 + T0m.s2*T0m.s2 + T0m.s3*T0m.s3);
           real T00 = T0m.s0;
           real ed_found;
           rootFinding(&ed_found, T00, M, eos_table);
           result[gid] = ed_found;
      }
    }
    """
    compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
    compile_options.append('-D USE_SINGLE_PRECISION')
    compile_options.append('-D EOSI')

    eos_table = Eos(cfg).create_table(self.ctx, compile_options)

    prg = cl.Program(self.ctx, kernel_src ).build(compile_options)
   
    size = 205*205*85
    edv = np.empty((size, 4), cfg.real)

    edv[:,0] = np.random.uniform(1.0, 10.0, size)
    v_mag = np.random.uniform(0.0, 0.999, size)
    theta = np.random.uniform(0.0, np.pi, size)
    phi = np.random.uniform(-np.pi, np.pi, size)
    edv[:,1] = v_mag * np.cos(theta) * np.cos(phi)
    edv[:,2] = v_mag * np.cos(theta) * np.sin(phi)
    edv[:,3] = v_mag * np.sin(theta)

    final = np.empty(size).astype(np.float32)
    mf = cl.mem_flags
    final_gpu = cl.Buffer(self.ctx, mf.READ_WRITE, final.nbytes)

    edv_gpu = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = edv)
    

    prg.rootfinding_test(self.queue, (size,), None,
                         edv_gpu, final_gpu, np.int32(size),
                         eos_table)
    
    cl.enqueue_copy(self.queue, final, final_gpu).wait()

    np.testing.assert_almost_equal(final, edv[:,0], 2)

    print('rootfinding test pass')



if __name__ == '__main__':
    unittest.main()
