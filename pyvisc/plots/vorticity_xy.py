#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Di 10 Mai 2016 11:14:30 CEST

import matplotlib.pyplot as plt
import numpy as np
import h5py
from HelmholtzHodge import HelmholtzHodge2D
import colormaps as cmap
import scipy.fftpack

class BulkPlot(object):
    def __init__(self, fname):
        self.h5 = h5py.File(fname, 'r')
        self.x = self.h5['coord/x']
        self.y = self.h5['coord/y']
        self.z = self.h5['coord/etas']

    def plot_xy(self, tau=4.0, scale=20, kind='gradient_free', color='k'):
        fed = ('bulk2d/exy_tau%s'%tau).replace('.', 'p')
        fvx = ('bulk2d/vx_xy_tau%s'%tau).replace('.', 'p')
        fvy = ('bulk2d/vy_xy_tau%s'%tau).replace('.', 'p')

        ed = self.h5[fed][...]
        vx = self.h5[fvx][...]
        vy = self.h5[fvy][...]
        x, y = self.x, self.y
        extent = [x[0], x[-1], y[0], y[-1]]

        mask = ed < 0.3
        ed[mask] = 0.0
        plt.imshow(ed.T, origin='lower', alpha=0.9,
               extent=extent, cmap=cmap.viridis)

        skip = 4
        hh = HelmholtzHodge2D(vx[::skip, ::skip],
                              vy[::skip, ::skip],
                               x[::skip],
                               y[::skip],
                              ed[::skip, ::skip])

        if kind == 'curl_free':
            hh.curl_free(scale=scale, color=color)
            plt.title(r'$curl\ free$')
        elif kind == 'gradient_free':
            hh.gradient_free(scale=scale, color=color)
            plt.title(r'$gradient\ free$')
        elif kind == 'full_vector':
            hh.quiver(scale=10, color=color)
            plt.title(r'$\mathbf{v}=(v_x, v_y)$')

        plt.xlabel(r'$x\ [fm]$')
        plt.ylabel(r'$y\ [fm]$')

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.subplots_adjust(bottom=0.15)


        #plt.show()

    def plot_zx(self, tau=4.0, scale=20, kind='gradient_free', color='k', ploted=True):
        fed = ('bulk2d/exz_tau%s'%tau).replace('.', 'p')
        fvx = ('bulk2d/vx_xz_tau%s'%tau).replace('.', 'p')
        fvz = ('bulk2d/vz_xz_tau%s'%tau).replace('.', 'p')

        ed = self.h5[fed][...]
        vx = self.h5[fvx][...]
        vz = self.h5[fvz][...]
        x, z = self.x, self.z
        extent = [z[0], z[-1], x[0], x[-1]]

        mask = ed < 0.3
        ed[mask] = 0.0

        if ploted:
            plt.imshow(ed, origin='lower', alpha=0.9,
               extent=extent, cmap=cmap.viridis)

        skip = 4
        hh = HelmholtzHodge2D(vz[::skip, ::skip].T,
                              vx[::skip, ::skip].T,
                               z[::skip],
                               x[::skip],
                              ed[::skip, ::skip].T)

        #hh return  parital_z vx, - partial_x vz, need flip for gradient_free

        if kind == 'curl_free':
            hh.curl_free(scale=scale, color=color)
            plt.title(r'$curl\ free$')
        elif kind == 'gradient_free':
            hh.gradient_free(scale=scale, color=color)
            plt.title(r'$gradient\ free$')
        elif kind == 'full_vector':
            hh.quiver(scale=10, color=color)
            plt.title(r'$\mathbf{v}=(v_{\eta}, v_x)$')

        plt.xlim(-9, 9)
        plt.ylim(-7, 7)
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$x\ [fm]$')
        plt.subplots_adjust(bottom=0.15)

        #plt.show()


if __name__=='__main__':
    bulk = BulkPlot('bulkinfo.h5')
    tau = 0.4 + 0.3 * 8 
    #bulk.plot_xy(tau, scale=1, kind='curl_free', color='k')
    #bulk.plot_xy(tau, scale=1, kind='full_vector', color='k')
    #bulk.plot_xy(tau, scale=0.5, kind='gradient_free', color='r')
    bulk.plot_zx(tau, scale=0.5, kind='full_vector', color='k')
    bulk.plot_zx(tau, scale=0.5, kind='gradient_free', color='r', ploted=False)

    plt.show()

