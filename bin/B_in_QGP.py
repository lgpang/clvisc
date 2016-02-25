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
    def __init__(self, eB0, sigx, sigy, hydro_cfg, bulkinfo=None, sigma=5.8):
        '''eB0: maximum magnetic field
           sigx: gaussian width of magnetic field along x
           sigy: gaussian width of magnetic field along y
           nx, ny: grids along x and y direction
           dx, dy: space step along x and y direction
           hydro_dir: directory with fluid velocity profile
           sigma: electric conductivity in units of MeV'''
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

        #By0 = eB0 * np.exp(-x*x/(2*sigx*sigx)-y*y/(2*sigy*sigy))
        By0 = eB0 * sigx*sigx/(x*x+sigx*sigx) * np.exp(-y*y/(2*sigy*sigy))
        Bx0 = eB0 * sigx*np.arctan(x/sigx) * y/(sigy*sigy) * np.exp(-y*y/(2*sigy*sigy))
        #Bx0 = np.zeros_like(By0)
        Bz0 = np.zeros_like(By0)

        self.B0 = [Bx0, By0, Bz0]

        #converts MeV to fm^{-1}
        self.sigma = sigma * 0.001 / 0.19732

        self.hydro_cfg = hydro_cfg
        self.bulkinfo = bulkinfo

        # self.B stores [[Bx, By, Bz], ...] array
        self.B = []

    def v_cross_B(self, v, B):
        ''' v cross B '''
        a = v[1]*B[2] - v[2]*B[1]
        b = -v[0]*B[2] + v[2]*B[0]
        c = v[0]*B[1] - v[1]*B[0]
        return np.array([a, b, c])

    def curl(self, B):
        ''' return: nabla X B '''
        dBx = np.gradient(B[0])
        dBy = np.gradient(B[1])
        dBz = np.gradient(B[2])
        zeros = np.zeros_like(B[0])
        dyBz, dzBy = dBz[1]/self.dy, zeros
        dxBz, dzBx = dBz[0]/self.dx, zeros
        dxBy, dyBx = dBy[0]/self.dx, dBx[1]/self.dy
        curl_x = dyBz - dzBy
        curl_y = dzBx - dxBz
        curl_z = dxBy - dyBx
        return np.array([curl_x, curl_y, curl_z])

    def electric_field(self, E0, v0, B0, v1, B1, dt, sigma):
        '''No RungeKutta: E^n+1 = (En + dt*(curl B - sigma vXB))/(1+sigma dt)
        with RungeKutta: E^n+1 = (En + 0.5dt*(curl B0 + J0 + curl B1 - sigma vxB))
        /(1+0.5*sigma*dt)'''
        curl_B0 = self.curl(B0)
        curl_B1 = self.curl(B1)
        v0_cross_B0 = self.v_cross_B(v0, B0)
        v1_cross_B1 = self.v_cross_B(v1, B1)
        def E_RungeKutta(i):
            J0i = sigma*(E0[i] + v0_cross_B0[i])
            return (E0[i] + 0.5*dt*(curl_B0[i] - J0i + curl_B1[i]
                    - sigma*v1_cross_B1[i]))/(1+0.5*sigma*dt)
        Ex = E_RungeKutta(0)
        Ey = E_RungeKutta(1)
        Ez = E_RungeKutta(2)
        return np.array([Ex, Ey, Ez])


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
        return np.array([vx, vy, vz])

    def check_divB(self, Bfield):
        dxBx = np.gradient(Bfield[0])[0]/self.dx
        dyBy = np.gradient(Bfield[1])[1]/self.dy
        err  = dxBx + dyBy
        return err


    def evolve(self, nstep=20):
        '''time evolution of magnetic field for nstep '''
        Bx = np.empty_like(self.B0[0])
        By = np.empty_like(Bx)
        Bz = np.empty_like(Bx)
        Bold = np.array(self.B0)
        zeros = np.zeros_like(Bx)

        Ex = np.zeros_like(Bx)
        Ey = np.zeros_like(Bx)
        Ez = np.zeros_like(Bx)
        Eold = np.array([Ex, Ey, Ez])

        eos_type = 'EOSI'
        if self.hydro_cfg.IEOS == 1:
            eos_type = 'EOSL'

        By_cent_vs_time = []

        extent = (self.x[0], self.x[-1], self.y[0], self.y[-1])

        sigma = self.sigma
        dt = self.dt

        for n in range(1, nstep):
            # predict step, get B^{n+1'} from B^n and E^n
            v0 = self.velocity(n-1)
            E0 = Eold
            curl_E0 = self.curl(E0)
            Bprim = Bold - dt * curl_E0

            v1 = self.velocity(n)
            #E1 = self.electric_field(E0, v0, Bold, v1, Bprim, dt=self.dt, sigma=self.sigma)
            Eprim = (E0 + dt*(self.curl(Bprim) - sigma*self.v_cross_B(v1, Bprim)))/(1+sigma*dt)

            B1 = Bold - 0.5*dt*(self.curl(E0) + self.curl(Eprim))

            Bold = B1
            Eold = (E0 + dt*(self.curl(B1) - sigma*self.v_cross_B(v1, B1)))/(1+sigma*dt)

            time = self.hydro_cfg.TAU0+n*dt
            divB = self.check_divB(Bold)
            plt.contourf(divB.T, origin='lower', extent=extent)
            plt.xlabel(r'$x\ [fm]$')
            plt.ylabel(r'$y\ [fm]$')
            plt.title(r'$div B\ @\ t=%s\ [fm]\ %s$'%(time, eos_type))
            smash_style.set()
            plt.colorbar()
            plt.savefig('%s/divB_%03d.png'%(self.hydro_dir,n))
            plt.close()

            self.B.append(Bold)

            i_cent = self.hydro_cfg.NX//2
            j_cent = self.hydro_cfg.NY//2
            By_cent = Bold[1, i_cent, j_cent]
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






def eB(eos_type='EOSL', sigma=5.8):
    '''sigma: electric conductivity'''
    if eos_type == 'EOSI':
        cfg.IEOS = 0
    else:
        cfg.IEOS = 1

    fout = '%s_figs_BW_sig%s'%(eos_type, sigma)
    fout = fout.replace('\.', 'p')

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

    eB_field = MagneticField(eB0=0.1, sigx=2.4, sigy=4.8, hydro_cfg=cfg, bulkinfo=bulk, sigma=sigma)

    eB_field.evolve(nstep=240)

    eB_field.plot()


if __name__=='__main__':
    #eB('EOSL', sigma=0)
    #eB('EOSL', sigma=5.8)
    #eB('EOSL', sigma=10000000)
    #eB('EOSL', sigma=20)
    eB('EOSL', sigma=100)
    #eB('EOSI', sigma=5.8)
