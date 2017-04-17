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
    def __init__(self, vx, vy, x, y):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.dx = x[1] - x[0]
        self.grad_v()

    def quiver(self):
        plt.quiver(self.x, self.y, self.vx.T, self.vy.T, scale=20)
        plt.xlabel(r'$x\ [fm]$')
        plt.ylabel(r'$y\ [fm]$')
        plt.title(r'$\mathbf{v}=(v_x, v_y)$')
        smash_style.set()
        plt.show()

    def grad_v(self):
        dvx = np.gradient(self.vx)
        dvy = np.gradient(self.vy)
        self.dvx = dvx/self.dx
        self.dvy = dvy/self.dx

    #@jit
    def int_nabla_v(self):
        ''' \vec{A}(r) = \int dr' curl v(r') / (r-r') dr'
        '''
        N = len(self.x)
        x, y = np.meshgrid(self.x, self.y)
        minus_dyvx = -self.dvx[1]
        dxvy = self.dvy[0]
        Ax = np.zeros_like(dxvy)
        Ay = np.zeros_like(dxvy)
        for i in range(N):
            for j in range(N):
                delta_x = x - (i-N/2)*self.dx
                delta_y = y - (j-N/2)*self.dx
                delta_r = np.sqrt(delta_x*delta_x + delta_y*delta_y)
                delta_r[delta_r<0.001] = 0.001
                Ax[i, j] = (dxvy/delta_r).sum()
                Ay[i, j] = (minus_dyvx/delta_r).sum()

        dxdy = self.dx**2.0
        return Ax*dxdy, Ay*dxdy

    def gradient_free(self):
        ''' nabla \times \vec{A} '''
        Ax, Ay = self.int_nabla_v()
        dyAx = np.gradient(Ax)[1]
        dxAy = np.gradient(Ay)[0]
        plt.xlabel(r'$x\ [fm]$')
        plt.ylabel(r'$y\ [fm]$')
        plt.quiver(self.x, self.y, dyAx.T, dxAy.T, scale=2000)
        plt.title(r'$\nabla\times \mathbf{a}(\mathbf{r})$')
        smash_style.set()
        plt.show()

    def curl_free(self):
        ''' A(r) = \int dr' grad \cdot v(r') / (r-r') dr'
        return grad A(r)
        '''
        N = len(self.x)
        x, y = np.meshgrid(self.x, self.y)
        dxvx = self.dvx[0]
        dyvy = self.dvy[1]
        Ar = np.zeros_like(dxvx)
        for i in range(N):
            for j in range(N):
                delta_x = x - (i-N/2)*self.dx
                delta_y = y - (j-N/2)*self.dx
                delta_r = np.sqrt(delta_x*delta_x + delta_y*delta_y)
                delta_r[delta_r<0.001] = 0.001
                Ar[i, j] = (dxvx/delta_r + dyvy/delta_r).sum()

        dxdy = self.dx**2.0
        Ar = -Ar*dxdy/(4*np.pi)

        grad_A = np.gradient(Ar)
        dxA, dyA = grad_A[0], grad_A[1]
        plt.quiver(self.x, self.y, dxA.T, dyA.T, scale=300)
        plt.xlabel(r'$x\ [fm]$')
        plt.ylabel(r'$y\ [fm]$')
        plt.title(r'$-\nabla\phi(\mathbf{r})$')
        smash_style.set()
        plt.show()


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
