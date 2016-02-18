#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Mi 17 Feb 2016 16:18:51 CET
''' calc the magnetic reponse of the QGP
with fluid velocity given by hydrodynamic simulations'''

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from math import floor
#import pyopencl as cl

import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg
#from visc import CLVisc
from ideal import CLIdeal

from common_plotting import smash_style

class MagneticField(object):
    def __init__(self, eB0, sigx, sigy, nx, ny, dx, dy, dt, hydro_dir, bulkinfo=None):
        '''eB0: maximum magnetic field
           sigx: gaussian width of magnetic field along x
           sigy: gaussian width of magnetic field along y
           nx, ny: grids along x and y direction
           dx, dy: space step along x and y direction
           hydro_dir: directory with fluid velocity profile
        '''
        self.hydro_dir = hydro_dir
        x = np.linspace(-floor(nx/2)*dx, floor(nx/2)*dx, nx, endpoint=True)
        y = np.linspace(-floor(ny/2)*dy, floor(ny/2)*dy, ny, endpoint=True)
        self.x = x
        self.y = y

        # for gradients and dB/dt calculation
        self.dx, self.dy, self.dt = dx, dy, dt

        x, y = np.meshgrid(x, y, indexing='ij')

        By0 = eB0 * np.exp(-x*x/(2*sigx*sigx)-y*y/(2*sigy*sigy))
        Bx0 = np.zeros_like(By0)
        Bz0 = np.zeros_like(By0)
        self.B0 = [Bx0, By0, Bz0]
        self.bulkinfo = bulkinfo

    def E(self, v, B):
        ''' E = - v cross B '''
        Ex = -v[1]*B[2] + v[2]*B[1]
        Ey = v[0]*B[2] - v[2]*B[0]
        Ez = -v[0]*B[1] + v[1]*B[0]
        return [Ex, Ey, Ez]

    def velocity(self, timestep):
        if self.bulkinfo == None:
            fvx = '%s/vx_xy%s.dat'%(self.hydro_dir, timestep)
            fvy = '%s/vy_xy%s.dat'%(self.hydro_dir, timestep)
            vx = np.loadtxt(fvx)
            vy = np.loadtxt(fvy)
        else:
            vx = self.bulkinfo.vx_xy[timestep]
            vy = self.bulkinfo.vy_xy[timestep]
        vz = np.zeros_like(vx)
        return [vx, vy, vz]


    def B(self, nstep=20):
        '''magnetic field after timestep update'''
        Bx = np.empty_like(self.B0[0])
        By = np.empty_like(Bx)
        Bz = np.empty_like(Bx)
        Bold = self.B0
        ax = self.dt / self.dx
        ay = self.dt / self.dy
        extent = (self.x[0], self.x[-1], self.y[0], self.y[-1])

        for n in range(1, nstep):
            # predict step, get B^{n+1'} from B^n and v^n
            v = self.velocity(n-1)
            E = self.E(v, Bold)
            dEx = np.gradient(E[0])
            dEy = np.gradient(E[1])
            dEz = np.gradient(E[2])
            dE = [dEx, dEy, dEz]

            Bx = Bold[0] - ay*dE[2][1]
            By = Bold[1] + ax*dE[2][0]
            Bz = Bold[2] - ax*dE[1][0] + ay*dE[0][1]
            Bprim = [Bx, By, Bz]

            v = self.velocity(n)
            E = self.E(v, Bprim)
            dEx = np.gradient(E[0])
            dEy = np.gradient(E[1])
            dEz = np.gradient(E[2])
            dE_prim = [dEx, dEy, dEz]

            Bx = Bold[0] - 0.5*ay*(dE[2][1] + dE_prim[2][1])
            By = Bold[1] + 0.5*ax*(dE[2][0] + dE_prim[2][0])
            Bz = 0.0
            Bold = [Bx, By, Bz]

            plt.contourf(Bold[1].T, origin='lower', extent=extent)
            #plt.imshow(Bold[1].T, extent=extent, vmin=0, vmax=0.1)
            plt.xlabel(r'$x\ [fm]$')
            plt.ylabel(r'$y\ [fm]$')
            plt.title(r'$B^{y}\ [GeV^2]\ @\ t=%s\ [fm]$'%(0.4+n*0.04))
            smash_style.set()
            plt.colorbar()
            plt.savefig('figs/BY%03d.png'%n)
            plt.close()

            #plt.contourf(Bold[0].T)
            plt.contourf(Bold[0].T, origin='lower', extent=extent)
            #plt.imshow(Bold[0].T, extent=extent, vmin=0, vmax=0.1)
            plt.xlabel(r'$x\ [fm]$')
            plt.ylabel(r'$y\ [fm]$')
            plt.title(r'$B^{x}\ [GeV^2]\ @\ t=%s\ [fm]$'%(0.4+n*0.04))
            smash_style.set()
            plt.colorbar()
            plt.savefig('figs/BX%03d.png'%n)
            plt.close()


        self.Bold = Bold



def eB_at(timestep=1):
    NX, NY = 301, 301
    DX, DY = 0.08, 0.08
    tau0, dt = 0.4, 0.3
    #hydro_dir = '/u/lpang/Magnetohydrodynamics/PyVisc/results/WBEOS_dNdEta_Events/squeezing_auau200_tau0p4_td1p9_eb0p00/'
    hydro_dir = '/u/lpang/PyVisc/results/D0/etaos_0p16/'
    eB_field = MagneticField(1.0, 1.3, 2.6, NX, NY, DX, DY, dt, hydro_dir)

    eB_field.B(nstep=timestep)
    plt.contourf(eB_field.Bold[0].T)
    plt.colorbar()
    plt.savefig('figs/eB%s.pdf'%timestep)
    plt.close()
    #plt.show()


def eB(fout):
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 401
    cfg.NY = 401
    cfg.NZ = 1

    cfg.DT = 0.005
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.DZ = 0.08
    cfg.IEOS = 0
    cfg.ntskip = 8

    cfg.ImpactParameter = 7.8
    cfg.Edmax = 55.0
    cfg.TAU0 = 0.4
    cfg.ETAOS = 0.08
    cfg.fPathOut = fout

    ideal = CLIdeal(cfg, gpu_id=2)
    from glauber import Glauber
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines, ideal.d_ev[1])
    ideal.evolve(max_loops=2000, to_maxloop=True, save_bulk=False,
                plot_bulk=True, save_hypersf=False)

    bulk = ideal.bulkinfo

    eB_field = MagneticField(0.1, 2.4, 4.8, cfg.NX, cfg.NY, cfg.DX, cfg.DY, dt=0.04,
            hydro_dir=fout, bulkinfo=bulk)
    eB_field.B(nstep=240)



if __name__=='__main__':
    eB('test_figs')
