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
from common_plotting import smash_style

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from ideal import CLIdeal
from config import cfg
from visc import CLVisc

class TestBjorken(unittest.TestCase):
    def setUp(self):
        cfg.NX = 8
        cfg.NY = 8
        cfg.NZ = 1
        cfg.IEOS = 0
        cfg.ETAOS = 0.08
        self.visc = CLVisc(cfg)
        self.ctx = self.visc.ideal.ctx
        self.queue = self.visc.ideal.queue


    def test_bjorken(self):
        ''' initialize with uniform energy density in (tau, x, y, eta) coordinates
        to test the Bjorken expansion:
           eps/eps0 = (tau/tau0)**(-4.0/3.0)
        '''

        kernel_src = """
        # include "real_type.h"
        //# include "eos_table.h"
        __kernel void init_ev(global real4 * d_ev1,
         //          global real * d_pi1,
                   read_only image2d_t eos_table,
                   const int size) {
          int gid = (int) get_global_id(0);
          if ( gid < size ) {
             d_ev1[gid] = (real4)(30.0f, 0.0f, 0.0f, 0.0f);
             //real S0 = S(30.0f, eos_table);
             //d_pi1[10*gid+9] = -4.0/3.0*ETAOS*S0/TAU0;
          }
        }
        """
        cwd, cwf = os.path.split(__file__)
    
        compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
        compile_options.append('-D USE_SINGLE_PRECISION')
        compile_options.append('-D ETAOS=%sf'%cfg.ETAOS)
        compile_options.append('-D TAU0=%sf'%cfg.TAU0)
        compile_options.append('-D S0=%sf'%self.visc.ideal.eos.f_S(30.0))
        print(compile_options)

        prg = cl.Program(self.ctx, kernel_src).build(compile_options)
        prg.init_ev(self.queue, (self.visc.ideal.size,), None,
                self.visc.ideal.d_ev[1], 
                self.visc.eos_table, np.int32(self.visc.ideal.size)).wait()

        self.visc.evolve(max_loops=2000, save_bulk=False,
                          save_hypersf=False)

        history = np.array(self.visc.ideal.history)
        tau, edmax = history[:,0], history[:,1]
        a = tau[0]/tau
        T0 = (edmax[0])**0.25
        lhs = edmax**0.25/T0
        b = cfg.ETAOS/tau[0]/0.36*0.19732*(1.0-a**(2.0/3.0))
        rhs = a**(1.0/3.0) * (1+2.0/3.0*b)

        import matplotlib.pyplot as plt
        plt.plot(tau-tau[0], lhs, 'r-', label='CLVisc')
        plt.plot(tau-tau[0], rhs, 'b--', label='Bjorken')
        plt.text(2., 0.9, r'$\tau_0=0.6\ fm$')
        plt.text(2., 0.82, r'$T_0=0.36\ GeV$')
        plt.text(2., 0.74, r'$\eta/s=0.08$')

        plt.xlabel(r'$\tau - \tau_0\ [fm]$')
        plt.ylabel(r'$T/T_0$')
        smash_style.set()
        plt.legend(loc='best')
        #plt.show()
        plt.savefig('bjorken_visc.pdf')
        #np.testing.assert_almost_equal(lhs, rhs, 2)
    

if __name__ == '__main__':
    unittest.main()
