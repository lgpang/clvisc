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


class MagneticField(object):
    def __init__(self, eB0, sigx, sigy, nx, ny, dx, dy, hydro_dir):
        '''eB0: maximum magnetic field
           sigx: gaussian width of magnetic field along x
           sigy: gaussian width of magnetic field along y
           nx, ny: grids along x and y direction
           dx, dy: space step along x and y direction
           hydro_dir: directory with fluid velocity profile
        '''
        self.hydro_dir = hydro_dir
        x = np.linspace(-floor(NX/2)*DX, floor(NX/2)*DX, NX, endpoint=True)
        y = np.linspace(-floor(NY/2)*DY, floor(NY/2)*DY, NY, endpoint=True)
        self.x = x
        self.y = y

        x, y = np.meshgrid(x, y, indexing='ij')

        By0 = eB0 * np.exp(-x*x/(2*sigx*sigx)-y*y/(2*sigy*sigy))
        Bx0 = np.zeros_like(By0)
        Bz0 = np.zeros_like(By0)
        self.B0 = [Bx0, By0, Bz0]

    def E(self, v, B):
        ''' E = - v cross B
        Notice we need nabla_z v_x, nabla_z v_y later'''
        Ex = v[1]*B[2] - v[2]*B[1]
        Ey = -v[0]*B[2] + v[2]*B[0]
        Ez = v[0]*B[1] - v[1]*B[0]
        return [Ex, Ey, Ez]

    def velocity(self, timestep):
        fvx = '%s/vx_xy%s.dat'%(self.hydro_dir, timestep)
        fvy = '%s/vy_xy%s.dat'%(self.hydro_dir, timestep)
        vx = np.loadtxt(fvx)
        vy = np.loadtxt(fvy)
        vz = np.zeros_like(vx)
        return [vx, vy, vz]


if __name__=='__main__':
    NX, NY = 401, 401
    DX, DY = 0.08, 0.08
    tau0, dt = 0.4, 0.3
    hydro_dir = '/u/lpang/Magnetohydrodynamics/PyVisc/results/WBEOS_dNdEta_Events/squeezing_auau200_tau0p4_td1p9_eb0p00/'
    eB_field = MagneticField(0.09, 2.4, 4.8, NX, NY, DX, DY, hydro_dir)
    plt.imshow(eB_field.B0[1])
    plt.show()

    
    #vx = np.loadtxt('%s/vx_xy0.dat'%hydro_dir)



