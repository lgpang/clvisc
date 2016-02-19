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
from config import write_config
#from visc import CLVisc
from ideal import CLIdeal

from common_plotting import smash_style

class MagneticField(object):
    def __init__(self, eB0, sigx, sigy, hydro_cfg, bulkinfo=None):
        '''eB0: maximum magnetic field
           sigx: gaussian width of magnetic field along x
           sigy: gaussian width of magnetic field along y
           nx, ny: grids along x and y direction
           dx, dy: space step along x and y direction
           hydro_dir: directory with fluid velocity profile '''
        nx, ny = hydro_cfg.NX, hydro_cfg.NY
        dx, dy = hydro_cfg.DX, hydro_cfg.DY
        dt = hydro_cfg.DT * hydro_cfg.ntskip
        self.hydro_dir = hydro_cfg.fPathOut

        x = np.linspace(-floor(nx/2)*dx, floor(nx/2)*dx, nx, endpoint=True)
        y = np.linspace(-floor(ny/2)*dy, floor(ny/2)*dy, ny, endpoint=True)
        self.x, self.y = x, y

        # for gradients and dB/dt calculation
        self.dx, self.dy, self.dt = dx, dy, dt

        x, y = np.meshgrid(x, y, indexing='ij')

        By0 = eB0 * np.exp(-x*x/(2*sigx*sigx)-y*y/(2*sigy*sigy))
        Bx0 = np.zeros_like(By0)
        Bz0 = np.zeros_like(By0)
        self.B0 = [Bx0, By0, Bz0]

        self.hydro_cfg = hydro_cfg
        self.bulkinfo = bulkinfo

        # self.B stores [[Bx, By, Bz], ...] array
        self.B = []

    def E(self, v, B):
        ''' E = - v cross B '''
        Ex = -v[1]*B[2] + v[2]*B[1]
        Ey = v[0]*B[2] - v[2]*B[0]
        Ez = -v[0]*B[1] + v[1]*B[0]
        return [Ex, Ey, Ez]

    def velocity(self, timestep):
        '''read fluid velocity from hydro_dir if
        ideal.evolve(save_bulk=True)
        or read fluid velocity from bulkinfo if
        ideal.evolve(plot_bulk=True) '''
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


    def evolve(self, nstep=20):
        '''time evolution of magnetic field for nstep '''
        Bx = np.empty_like(self.B0[0])
        By = np.empty_like(Bx)
        Bz = np.empty_like(Bx)
        Bold = self.B0
        ax = self.dt / self.dx
        ay = self.dt / self.dy

        eos_type = 'EOSI'
        if self.hydro_cfg.IEOS == 1:
            eos_type = 'EOSL'

        By_cent_vs_time = []

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
            dEx_1 = np.gradient(E[0])
            dEy_1 = np.gradient(E[1])
            dEz_1 = np.gradient(E[2])
            dE_prim = [dEx_1, dEy_1, dEz_1]

            Bx = Bold[0] - 0.5*ay*(dE[2][1] + dE_prim[2][1])
            By = Bold[1] + 0.5*ax*(dE[2][0] + dE_prim[2][0])
            Bz = 0.0
            Bold = [Bx, By, Bz]

            self.B.append(Bold)

            time = self.hydro_cfg.TAU0+n*self.dt 
            i_cent = self.hydro_cfg.NX//2
            j_cent = self.hydro_cfg.NY//2
            By_cent = By[i_cent, j_cent]
            By_cent_vs_time.append([time, By_cent])

        By_cent_vs_time = np.array(By_cent_vs_time)
        fname = '%s/By_cent_vs_time_%s.dat'%(self.hydro_dir, eos_type)
        np.savetxt(fname, By_cent_vs_time, header='tau, By(x=0, y=0) GeV^2')
        print('evolution finished! start to plot...')


    def plot(self, ntskip=1):
        extent = (self.x[0], self.x[-1], self.y[0], self.y[-1])
        eos_type = 'EOSI'
        if self.hydro_cfg.IEOS == 1:
            eos_type = 'EOSL'
        for n, Bold in enumerate(self.B):
            time = self.hydro_cfg.TAU0+n*self.dt 
            plt.contourf(Bold[1].T, origin='lower', extent=extent)
            plt.xlabel(r'$x\ [fm]$')
            plt.ylabel(r'$y\ [fm]$')
            plt.title(r'$B^{y}\ [GeV^2]\ @\ t=%s\ [fm]\ %s$'%(time, eos_type))
            smash_style.set()
            plt.colorbar()
            plt.savefig('%s/BY%03d.png'%(self.hydro_dir,n))
            plt.close()

            #plt.contourf(Bold[0].T)
            plt.contourf(Bold[0].T, origin='lower', extent=extent)
            #plt.imshow(Bold[0].T, extent=extent, vmin=0, vmax=0.1)
            plt.xlabel(r'$x\ [fm]$')
            plt.ylabel(r'$y\ [fm]$')
            plt.title(r'$B^{x}\ [GeV^2]\ @\ t=%s\ [fm]\ %s$'%(time, eos_type))
            smash_style.set()
            plt.colorbar()
            plt.savefig('%s/BX%03d.png'%(self.hydro_dir, n))
            plt.close()






def eB(eos_type='EOSL'):
    if eos_type == 'EOSI':
        cfg.IEOS = 0
    else:
        cfg.IEOS = 1

    fout = '%s_figs'%eos_type

    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 401
    cfg.NY = 401
    cfg.NZ = 1

    cfg.DT = 0.005
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.DZ = 0.08
    cfg.ntskip = 8

    cfg.ImpactParameter = 7.8
    cfg.Edmax = 55.0
    cfg.TAU0 = 0.4
    cfg.ETAOS = 0.08
    cfg.fPathOut = fout

    write_config(cfg)

    ideal = CLIdeal(cfg, gpu_id=2)

    from glauber import Glauber
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines, ideal.d_ev[1])

    ideal.evolve(max_loops=2000, to_maxloop=True, save_bulk=False,
                plot_bulk=True, save_hypersf=False)

    bulk = ideal.bulkinfo

    eB_field = MagneticField(eB0=0.1, sigx=2.4, sigy=4.8, hydro_cfg=cfg, bulkinfo=bulk)

    eB_field.evolve(nstep=240)

    eB_field.plot()


if __name__=='__main__':
    eB('EOSL')
    eB('EOSI')
