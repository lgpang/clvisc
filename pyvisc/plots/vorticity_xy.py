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
    def __init__(self, fname, n=5):
        self.h5 = h5py.File(fname, 'r')
        self.x = self.h5['coord/x'][...]
        self.y = self.h5['coord/y'][...]
        self.z = self.h5['coord/etas'][...]
        self.nt = n
        tau0 =  self.h5['coord/tau'][...][0]
        tau1 =  self.h5['coord/tau'][...][1]
        self.tau = tau0 + n * (tau1 - tau0)

    def plot_xy(self, scale=20, kind='gradient_free', color='k', ploted=True):
        tau = self.tau
        fed = ('bulk2d/exy_tau%.1f'%tau).replace('.', 'p')
        fvx = ('bulk2d/vx_xy_tau%.1f'%tau).replace('.', 'p')
        fvy = ('bulk2d/vy_xy_tau%.1f'%tau).replace('.', 'p')

        ed = self.h5[fed][...]
        vx = self.h5[fvx][...]
        vy = self.h5[fvy][...]
        x, y = self.x, self.y
        extent = [x[0], x[-1], y[0], y[-1]]

        mask = ed < 0.3
        ed[mask] = 0.0
        image = None
        if ploted:
            image = plt.imshow(ed.T, origin='lower', alpha=0.9,
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
            plt.title(r'$divergence\ free\ \tau=%.1f\ fm$'%self.tau)
        elif kind == 'full_vector':
            hh.quiver(scale=10, color=color)
            plt.title(r'$\mathbf{v}=(v_x, v_y)$')

        plt.xlabel(r'$x\ [fm]$')
        plt.ylabel(r'$y\ [fm]$')

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.subplots_adjust(bottom=0.15)

        return image


        #plt.show()

    def plot_zx(self, scale=20, kind='gradient_free', color='k', ploted=True):
        tau = self.tau
        fed = ('bulk2d/exz_tau%.1f'%tau).replace('.', 'p')
        fvx = ('bulk2d/vx_xz_tau%.1f'%tau).replace('.', 'p')
        fvz = ('bulk2d/vz_xz_tau%.1f'%tau).replace('.', 'p')

        ed = self.h5[fed][...]
        vx = self.h5[fvx][...]
        vz = self.h5[fvz][...]
        x, z = self.x, self.z
        extent = [z[0], z[-1], x[0], x[-1]]

        mask = ed < 0.3
        ed[mask] = 0.0

        image = None
        if ploted:
            image = plt.imshow(ed, origin='lower', alpha=0.9,
               extent=extent, cmap=cmap.viridis)

        skip = 4

        gamma = 1.0/np.sqrt(1.0 - vx*vx - vz*vz)

        gamma = gamma[::skip, ::skip].T
        hh = HelmholtzHodge2D(gamma * vz[::skip, ::skip].T,
                              gamma * vx[::skip, ::skip].T,
                               z[::skip],
                               x[::skip],
                              ed[::skip, ::skip].T)

        #hh return  parital_z vx, - partial_x vz, need flip for gradient_free

        if kind == 'curl_free':
            hh.curl_free(scale=scale, color=color)
            plt.title(r'$curl\ free$')
        elif kind == 'gradient_free':
            hh.gradient_free(scale=scale, color=color, flip=True)
            plt.title(r'$divergence\ free\ \tau=%.1f\ fm$'%self.tau)
        elif kind == 'full_vector':
            hh.quiver(scale=10, color=color)
            plt.title(r'$\mathbf{v}=(v_{\eta}, v_x)$')

        plt.xlim(-9, 9)
        plt.ylim(-7, 7)
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$x\ [fm]$')
        plt.subplots_adjust(bottom=0.15)

        return image

        #plt.show()


def main(n=0):
    bulk_ideal = BulkPlot('ideal_bulkinfo.h5', n=n)
    bulk_visc  = BulkPlot('visc_bulkinfo.h5', n=n)
    tau = bulk_ideal.tau

    figure = plt.figure(figsize=(15, 8))
    plt.subplot(121)
    bulk_ideal.plot_xy(scale=0.3, kind='gradient_free', color='w', ploted=True)
    #bulk_ideal.plot_zx(scale=0.5, kind='gradient_free', color='w', ploted=True)
    plt.text(0.35, 1.05, r"$(a)\ \eta_v/s=0$", transform=plt.gca().transAxes, color='k', size=30)
    plt.text(0.85, 1.05, r"$\tau=%.1f\ fm$"%tau, transform=plt.gca().transAxes, color='k', size=40)
    plt.title("")

    plt.subplot(122)
    bulk_visc.plot_xy(scale=0.3, kind='gradient_free', color='w', ploted=True)
    #img = bulk_visc.plot_zx(scale=0.5, kind='gradient_free', color='w', ploted=True)
    plt.text(0.3, 1.05, r"$(b)\ \eta_v/s=0.08$", transform=plt.gca().transAxes, color='k', size=30)
    plt.ylabel("")
    plt.gca().set_yticks([])
    plt.title("")

    #plt.colorbar(img)

    plt.savefig('figs/cmp_hhxy%03d.png'%n)
    #plt.savefig('figs/cmp_hhxz%03d.png'%n)
    plt.show()

#main(10)

for n in range(20):
    main(n)
