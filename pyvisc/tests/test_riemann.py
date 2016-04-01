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
        cfg.TAU0 = 0.0
        cfg.ntskip = 100
        cfg.IEOS = 0
        cfg.ETAOS = 0.0
        cfg.save_to_hdf5 = False
        cfg.riemann_test = True


    def test_riemann(self):
        ''' initialize with step energy density in
        (t, x, y, z) coordinates to test the riemann solution,
        '''

        z = np.linspace(-10, 10, cfg.NZ)
        edv = np.zeros((cfg.NX*cfg.NY*cfg.NZ, 4), dtype=np.float32)

        from gubser import Riemann
        analytical_solution = Riemann(z, pressure_left=1.0)

        #cfg.TAU0 = 1.0
        self.visc = CLVisc(cfg)
        self.ctx = self.visc.ctx
        self.queue = self.visc.queue

        ed_ini = analytical_solution.energy_density(cfg.TAU0)
        vz_ini = analytical_solution.fluid_velocity(cfg.TAU0)

        for i in range(cfg.NX):
            for j in range(cfg.NY):
                for k in range(cfg.NZ):
                    index = i*cfg.NY*cfg.NZ + j*cfg.NZ + k
                    edv[index, 0] = ed_ini[k]
                    edv[index, 3] = vz_ini[k]
        
        self.visc.ideal.load_ini(edv)

        self.visc.evolve(max_loops=1001, plot_bulk=True, save_hypersf=False,
                         force_run_to_maxloop=True)

        bulk = self.visc.ideal.bulkinfo

        import matplotlib.pyplot as plt

        import h5py
        h5 = h5py.File('riemann_ideal.h5', 'w')
        h5.create_dataset('z', data=z)

        nstep = 10
        tau_list = np.empty(nstep)
        cs2 = 1.0/3.0
        for i in range(nstep):
            h5.create_dataset('clvisc/ed/%s'%i, data=bulk.ez[i])
            h5.create_dataset('clvisc/pr/%s'%i, data=bulk.ez[i]*cs2)
            h5.create_dataset('clvisc/vz/%s'%i, data=bulk.vz[i])

            h5.create_dataset('riemann/ed/%s'%i, data=
                    analytical_solution.energy_density(i))
            h5.create_dataset('riemann/pr/%s'%i, data=
                    analytical_solution.pressure(i))
            h5.create_dataset('riemann/vz/%s'%i, data=
                    analytical_solution.fluid_velocity(i))

            tau = cfg.TAU0 + i*cfg.ntskip*cfg.DT
            tau_list[i] = tau

        h5.create_dataset('tau', data=tau_list)
        h5.close()

        for i in [0,2,4,8]:
            plt.plot(z, bulk.ez[i]/3.0, '--')
            ez = analytical_solution.pressure(i)
            plt.plot(z, ez)
            plt.text(-i, 0.9, r'$\tau=%s\ [fm]$'%i)

        plt.ylim(-0.5, 1.5)
        plt.xlabel(r'$z\ [fm]$')
        plt.ylabel(r'$\varepsilon$')
        smash_style.set(line_styles=False)
        plt.savefig('riemann_ed.pdf')
        plt.close()

        for i in [0,2,4,8]:
            plt.plot(z, bulk.vz[i], '--')
            vz = analytical_solution.fluid_velocity(i)
            plt.plot(z, vz)
            plt.text(i, 0.9, r'$\tau=%s\ [fm]$'%i)

        plt.ylim(-0.5, 1.5)
        plt.xlabel(r'$z\ [fm]$')
        plt.ylabel(r'$v_z$')
        smash_style.set(line_styles=False)
        plt.savefig('riemann_vz.pdf')
        plt.close()


   

if __name__ == '__main__':
    unittest.main()
