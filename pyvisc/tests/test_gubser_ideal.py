#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import pyopencl as cl
from pyopencl import array
import os, sys
from time import time
import numpy as np
from scipy.special import hyp2f1
import unittest
import matplotlib.pyplot as plt

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from ideal import CLIdeal
from config import cfg


def gubser_ed(tau, r, q):
    return (2.0*q)**(8.0/3.0)/(1+2*q*q*(tau*tau+r*r)+q**4*(tau*tau-r*r)**2)**(4.0/3.0)

def gubser_vr(tau, r, q):
    return 2.0*q*q*tau*r/(1.0+q*q*tau*tau+q*q*r*r)

class TestGubser(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.ideal = CLIdeal(self.cfg)
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue

    def test_gubser(self):
        ''' initialize with gubser energy density in (tau, x, y, eta) coordinates
        to test the gubser expansion:
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
        a = (tau/tau[0])**(-4.0/3.0)
        b = edmax/edmax[0]
        np.testing.assert_almost_equal(a, b, 2)
    

if __name__ == '__main__':
    unittest.main()




def plot():
    tau_list = np.array( [1.0, 1.2, 1.4, 1.6, 2.0, 3.0 ] )
    x = np.linspace(-4, 4, 100)
    
    for tau in tau_list:
        y = gubser_ed(tau, x, 0.25)
        txt = plt.text(-0.5, y+0.01, "tau=%s"%tau, fontsize=20)
        plt.plot(x, y, 'k-')
    plt.legend( loc='best' )
    plt.xlabel( r'$r_T$ [fm]' )
    plt.ylabel( r'T' )
    plt.show()



