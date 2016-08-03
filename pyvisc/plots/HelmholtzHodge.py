#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 04 Feb 2016 5:38:06 CET
'''Helmholtz-Hodge vector fields decomposition
   For any vector fields \vec{v}, it can be
   decomposed to curl-free and divergence-free part;
   This library is used to extract the vorticity developped
   during the QGP expansion in relativistic high energy heavy ion collisions,
   to visualize how does the initial angular momentum transfers to
   local vorticity'''

import matplotlib.pyplot as plt
import numpy as np
#from numba import jit
from common_plotting import smash_style


class HelmholtzHodge2D(object):
    def __init__(self, vx, vy, x, y, ed):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ed = ed
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.__grad_v()

    def __grad_v(self):
        # remember np.gradient() return (dy, dx)
        self.dvx = np.gradient(self.vx)
        self.dvy = np.gradient(self.vy)

    def quiver(self, scale=20, color='k'):
        vx = self.vx
        vy = self.vy
        mask = self.ed < 0.3
        vx[mask] = 0.0
        vy[mask] = 0.0
        plt.quiver(self.x, self.y, vx.T, vy.T,
                   scale=scale, color=color)
        smash_style.set()

    #@jit
    def __int_nabla_v(self):
        ''' \vec{A}(r) = \int dr' curl v(r') / (r-r') dr'
        '''
        Nx = len(self.x)
        Ny = len(self.y)
        x, y = np.meshgrid(self.x, self.y, indexing='ij')
        dyvx = self.dvx[1]/self.dy
        dxvy = self.dvy[0]/self.dx
        Az = np.zeros_like(dxvy)
        for i in range(Nx):
            for j in range(Ny):
                delta_x = x - (i-Nx/2)*self.dx
                delta_y = y - (j-Ny/2)*self.dy
                delta_r = np.sqrt(delta_x*delta_x + delta_y*delta_y)
                delta_r[delta_r<0.1] = 0.1
                Az[i, j] =  ((dxvy - dyvx)/delta_r).sum()

        dxdy = self.dx*self.dy
        return Az*dxdy/(np.pi*4)

    def gradient_free(self, scale=20, color='k'):
        ''' nabla \times \vec{A} '''
        Az = self.__int_nabla_v()
        x, y = np.meshgrid(self.x, self.y, indexing='ij')
        dyAz = np.gradient(Az)[1]
        dxAz = np.gradient(Az)[0]
        mask = self.ed < 0.3
        dyAz[mask] = 0.0
        dxAz[mask] = 0.0

        plt.quiver(x, y, dyAz, -dxAz, scale=scale, color=color)
        smash_style.set()

    def curl_free(self, scale=20, color='k'):
        ''' A(r) = \int dr' grad \cdot v(r') / (r-r') dr'
        return grad A(r)
        '''
        Nx = len(self.x)
        Ny = len(self.y)
        x, y = np.meshgrid(self.x, self.y, indexing='ij')
        dxvx = self.dvx[0]/self.dx
        dyvy = self.dvy[1]/self.dy
        Ar = np.zeros_like(dxvx)
        for i in range(Nx):
            for j in range(Ny):
                delta_x = x - (i-Nx/2)*self.dx
                delta_y = y - (j-Ny/2)*self.dy
                delta_r = np.sqrt(delta_x*delta_x + delta_y*delta_y)
                delta_r[delta_r<0.001] = 0.001
                Ar[i, j] = (dxvx/delta_r + dyvy/delta_r).sum()

        dxdy = self.dx * self.dy
        Ar = -Ar*dxdy/(4*np.pi)

        grad_A = np.gradient(Ar)
        dxA, dyA = grad_A[0], grad_A[1]
        mask = self.ed < 0.03
        dxA[mask] = 0.0
        dyA[mask] = 0.0
        plt.quiver(x, y, dxA, dyA,
                   scale=scale, color=color)

        smash_style.set()


def example():
    x = np.linspace(-15, 15, 301)
    y = np.linspace(-15, 15, 301)
    
    ''' notice that vx, vy should be stored in the following way
    to 2 different files.
    for ( int i=0; i<Nx; i++ ) {
        for ( int j=0; j<Ny; j++ ) {
            fout << vx[i][j] << ' ';
        }
        fout << std::endl;
    }
    '''
    
    #vx = np.loadtxt('../results/P30_idealgas/vx_xy8.dat')
    #vy = np.loadtxt('../results/P30_idealgas/vy_xy8.dat')
    
    vx = np.loadtxt('../results/P30/vx_xy8.dat')
    vy = np.loadtxt('../results/P30/vy_xy8.dat')
    
    hh = HelmholtzHodge2D(vx[::5, ::5], vy[::5, ::5], x[::5], y[::5])
    #hh = HelmholtzHodge2D(vx, vy, x, y)
    hh.quiver()
    #hh.gradient_free()
    #hh.curl_free()


if __name__ == '__main__':
    example()
