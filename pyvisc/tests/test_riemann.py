#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 31 Mar 2016 11:59:20 CEST

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
        cfg.NX = 1
        cfg.NY = 1
        cfg.NZ = 401
        cfg.DZ = 0.05
        cfg.DT = 0.01
        cfg.ntskip = 100
        cfg.IEOS = 0
        cfg.ETAOS = 0.0
        cfg.save_to_hdf5 = False
        cfg.riemann_test = True
        self.visc = CLVisc(cfg)
        self.ctx = self.visc.ctx
        self.queue = self.visc.queue


    def test_riemann(self):
        ''' initialize with step energy density in
        (t, x, y, z) coordinates to test the riemann solution,
        '''

        z = np.linspace(-10, 10, cfg.NZ)
        edv = np.zeros((cfg.NX*cfg.NY*cfg.NZ, 4), dtype=np.float32)

        for i in range(cfg.NX):
            for j in range(cfg.NY):
                for k in range(cfg.NZ):
                    index = i*cfg.NY*cfg.NZ + j*cfg.NZ + k
                    if k < cfg.NZ/2:
                        edv[index, 0] = 3.0
                    if k > cfg.NZ/2:
                        edv[index, 3] = 1.0
        
        self.visc.ideal.load_ini(edv)

        self.visc.evolve(max_loops=1001, plot_bulk=True, save_hypersf=False,
                         force_run_to_maxloop=True)

        bulk = self.visc.ideal.bulkinfo

        import matplotlib.pyplot as plt
        for i in range(10):
            plt.plot(z, bulk.ez[i]/3.0)
            #plt.plot(z, bulk.vz[i])

        plt.ylim(-0.5, 1.5)

        plt.show()

   

if __name__ == '__main__':
    unittest.main()
