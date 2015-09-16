import pyopencl as cl
import numpy as np
from time import time
import os

import unittest


class TestEos(unittest.TestCase):
  def test_eos(self):
    os.environ['PYOPENCL_CTX'] = ':1'
    ctx = cl.create_some_context()
    queue = cl.CommandQueue( ctx )
    mf = cl.mem_flags

    kernel_src = """
    #include "EosPCEv0.cl"
    
    __kernel void PTS(global real * eps, 
                      global real * pts, 
 		      const int size) {
      int gid = (int) get_global_id(0);
      if ( gid < size ) {
         real energy_density = eps[gid];
         pts[3*gid+0] = P(energy_density);
         pts[3*gid+1] = T(energy_density);
         pts[3*gid+2] = S(energy_density);
      }
    }
    """

    num_of_eps = 10000

    x = np.linspace(0.0, 30.0, num_of_eps, dtype=np.float32)

    x_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    cwd, cwf = os.path.split(__file__)

    compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
    #compile_options.append('-D EOSLPCE')
    compile_options.append('-D EOSLCE')
    compile_options.append('-D USE_SINGLE_PRECISION')

    prg = cl.Program(ctx, kernel_src).build(compile_options)

    globalsize = 3*num_of_eps

    pts = np.zeros(globalsize).astype( np.float32 )

    pts_gpu = cl.Buffer(ctx, mf.READ_WRITE, pts.nbytes)
    
    prg.PTS(queue, (globalsize,), None, x_gpu, pts_gpu, np.int32(num_of_eps))

    cl.enqueue_read_buffer(queue, pts_gpu, pts).wait()

    import matplotlib.pyplot as plt
    pts = np.array(pts).reshape(num_of_eps, 3)

    # dP/de in EOSLCE and EOSLPCE are not smooth for small ed
    eps = x[:-1]
    de = x[1:] - x[:-1]
    plt.plot(eps, np.diff(pts[:,0], 1)/de)
    plt.show()

    #print 'gpu results = ',  pts

    #self.assertAlmostEqual(xmax, final)


if __name__ == '__main__':
  unittest.main()
