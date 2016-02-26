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

def plot_bulk(bulk):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(10):
        ax[0].semilogy(bulk.z, bulk.ez[i])
        ax[1].plot(bulk.z, bulk.vz[i])

    ax[0].set_xlabel(r'$\eta_s$', fontsize=25)
    ax[0].set_ylabel(r'$\varepsilon$', fontsize=25)
    ax[0].set_ylim(1.0E-4, 1.0E2)

    plt.show()


def save_bulk(bulk, timestep):
    dat = []
    comments = 'etas, (ed, vz) as a function of etas for tau='\
             + ','.join(['%s'%(i*timestep+0.6) for i in  range(10)])

    dat.append(bulk.z)
    for i in range(10):
        dat.append(bulk.ez[i])
        dat.append(bulk.vz[i])
    np.savetxt('longitudinal_expansion_gaussian_dz0p075.dat', np.array(dat).T,
            fmt='%.4f', header=comments)
    


class TestLongitudinalExpansion(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.cfg.NX = 1
        self.cfg.NY = 1
        self.cfg.NZ = 401
        self.cfg.DT = 0.005
        self.cfg.DZ = 0.075
        self.cfg.IEOS = 0
        self.cfg.ntskip=100
        self.cfg.nzskip=4
        self.ideal = CLIdeal(self.cfg)
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue


    def test_longitudinal_expansion(self):
        ''' initialize with uniform energy density in (tau, x, y, eta) coordinates
        to test the Bjorken expansion:
           eps/eps0 = (tau/tau0)**(-4.0/3.0)
        '''

        kernel_src = """
        # include "real_type.h"
        # define Eta_flat 2.95f
        # define Eta_gw   2.0f
        # define NZ %s
        # define DZ %sf

        real weight_along_eta(real z, real etas0_) {
            real heta;
            if( fabs(z-etas0_) > Eta_flat ) {
                heta=exp(-pow(fabs(z-etas0_)-Eta_flat,2.0f)/(2.0f*Eta_gw*Eta_gw));
            } else {
                heta = 1.0f;
            }
            return heta;
        }

        real gaussian_along_eta(real z) {
            return exp(-z*z/(2.0f*Eta_gw*Eta_gw));
        }


        __kernel void init_ev(global real4 * d_ev1, const int size) {
          int gid = (int) get_global_id(0);
          if ( gid < size ) {
             real eta = (gid - NZ/2)*DZ;
             real eds = gaussian_along_eta(eta) * 30.0;
             d_ev1[gid] = (real4)(eds, 0.0f, 0.0f, 0.0f);
          }
        }
        """%(self.cfg.NZ, self.cfg.DZ)
        cwd, cwf = os.path.split(__file__)
    
        compile_options = ['-I %s'%os.path.join(cwd, '..', 'kernel')]
        compile_options.append('-D USE_SINGLE_PRECISION')
        prg = cl.Program(self.ctx, kernel_src).build(compile_options)
        prg.init_ev(self.queue, (self.ideal.size,), None, self.ideal.d_ev[1],
                    np.int32(self.ideal.size)).wait()

        self.ideal.evolve(max_loops=1400, save_bulk=False, save_hypersf=False,
                plot_bulk=True)

        bulk = self.ideal.bulkinfo

        save_bulk(bulk, timestep=self.cfg.DT*self.cfg.ntskip)

        #plot_bulk(bulk)



        #history = np.array(self.ideal.history)
        #tau, edmax = history[:,0], history[:,1]
        #a = (tau/tau[0])**(-4.0/3.0)
        #b = edmax/edmax[0]
        #np.testing.assert_almost_equal(a, b, 3)
    

if __name__ == '__main__':
    unittest.main()
