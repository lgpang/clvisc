#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 15 Oct 2015 14:02:44 CEST

from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time
from math import floor

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
from eos.eos import Eos
import h5py


class BulkInfo(object):
    '''The bulk information like:
       ed(x), ed(y), ed(eta), T(x), T(y), T(eta)
       vx, vy, veta, ecc_x, ecc_p'''
    def __init__(self, cfg, ctx, queue, compile_options):
        self.cfg = cfg
        self.ctx = ctx
        self.queue = queue
        self.compile_options = list(compile_options)

        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ

        self.x = np.linspace(-floor(NX/2)*cfg.DX, floor(NX/2)*cfg.DX, NX, endpoint=True)
        self.y = np.linspace(-floor(NY/2)*cfg.DY, floor(NY/2)*cfg.DY, NY, endpoint=True)
        self.z = np.linspace(-floor(NZ/2)*cfg.DZ, floor(NZ/2)*cfg.DZ, NZ, endpoint=True)

        self.h_ev = np.zeros((NX*NY*NZ, 4), cfg.real)
        
        # store the data in hdf5 file
        h5_path = os.path.join(cfg.fPathOut, 'bulkinfo.h5')
        self.f_hdf5 = h5py.File(h5_path, 'w')

        self.eos = Eos(cfg.IEOS)

        # time evolution for , edmax and ed, T at (x=0,y=0,etas=0)
        self.time = []
        self.edmax = []
        self.edcent = []
        self.Tcent = []

        # time evolution for total_entropy, eccp, eccx and <vr>
        self.entropy = []
        self.eccp_vs_tau = []
        self.eccx = []
        self.vr= []

    def get(self, tau, d_ev, edmax):
        ''' store the bulkinfo to hdf5 file '''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.time.append(tau)
        self.edmax.append(edmax)

        cl.enqueue_copy(self.queue, self.h_ev, d_ev).wait()
        bulk = self.h_ev.reshape(NX, NY, NZ, 4)

        # tau=0.6 changes to tau='0p6'
        time_stamp = ('%s'%tau).replace('.', 'p')

        i0, j0, k0 = NX//2, NY//2, NZ//2

        exy = bulk[:, :, k0, 0]
        vx = bulk[:, :, k0, 1]
        vy = bulk[:, :, k0, 2]

        self.eccp_vs_tau.append(self.eccp(exy, vx, vy)[1])
        self.vr.append(self.mean_vr(exy, vx, vy))
        self.entropy.append(self.total_entropy(tau, exy, vx, vy))

        ed_cent = exy[i0, j0]

        self.edcent.append(ed_cent)
        self.Tcent.append(self.eos.f_T(ed_cent))

        ecc1, ecc2 = self.ecc_vs_rapidity(bulk)
        self.f_hdf5.create_dataset('bulk1d/eccp1_tau%s'%time_stamp, data = ecc1)
        self.f_hdf5.create_dataset('bulk1d/eccp2_tau%s'%time_stamp, data = ecc2)

        # ed_x(y=0, z=0), ed_y(x=0, z=0), ed_z(x=0, y=0)
        self.f_hdf5.create_dataset('bulk1d/ex_tau%s'%time_stamp, data = bulk[:, j0, k0, 0])
        self.f_hdf5.create_dataset('bulk1d/ey_tau%s'%time_stamp, data = bulk[i0, :, k0, 0])
        self.f_hdf5.create_dataset('bulk1d/ez_tau%s'%time_stamp, data = bulk[i0, j0, :, 0])

        # vx_x(y=0, z=0), vy_y(x=0, z=0), vz_z(x=0, y=0)
        self.f_hdf5.create_dataset('bulk1d/vx_tau%s'%time_stamp, data = bulk[:, j0, k0, 1])
        self.f_hdf5.create_dataset('bulk1d/vy_tau%s'%time_stamp, data = bulk[i0, :, k0, 2])
        self.f_hdf5.create_dataset('bulk1d/vz_tau%s'%time_stamp, data = bulk[i0, j0, :, 3])

        # ed_xy(z=0), ed_xz(y=0), ed_yz(x=0)
        self.f_hdf5.create_dataset('bulk2d/exy_tau%s'%time_stamp, data = bulk[:, :, k0, 0])
        self.f_hdf5.create_dataset('bulk2d/exz_tau%s'%time_stamp, data = bulk[:, j0, :, 0])
        self.f_hdf5.create_dataset('bulk2d/eyz_tau%s'%time_stamp, data = bulk[i0, :, :, 0])

        # vx_xy(z=0), vx_xz(y=0), vx_yz(x=0)
        self.f_hdf5.create_dataset('bulk2d/vx_xy_tau%s'%time_stamp, data = bulk[:, :, k0, 1])
        self.f_hdf5.create_dataset('bulk2d/vx_xz_tau%s'%time_stamp, data = bulk[:, j0, :, 1])
        #self.f_hdf5.create_dataset('bulk2d/vx_yz_tau%s'%time_stamp, data = bulk[i0, :, :, 1])

        # vy_xy(z=0), vy_xz(y=0), vy_yz(x=0)
        self.f_hdf5.create_dataset('bulk2d/vy_xy_tau%s'%time_stamp, data = bulk[:, :, k0, 2])
        #self.f_hdf5.create_dataset('bulk2d/vy_xz_tau%s'%time_stamp, data = bulk[:, j0, :, 2])
        self.f_hdf5.create_dataset('bulk2d/vy_yz_tau%s'%time_stamp, data = bulk[i0, :, :, 2])

        # vz_xy(z=0), vz_xz(y=0), vz_yz(x=0)
        self.f_hdf5.create_dataset('bulk2d/vz_xy_tau%s'%time_stamp, data = bulk[:, :, k0, 3])
        self.f_hdf5.create_dataset('bulk2d/vz_xz_tau%s'%time_stamp, data = bulk[:, j0, :, 3])
        #self.f_hdf5.create_dataset('bulk2d/vz_yz_tau%s'%time_stamp, data = bulk[i0, :, :, 3])



    def eccp(self, ed, vx, vy, vz=0.0):
        ''' eccx = <y*y-x*x>/<y*y+x*x> where <> are averaged 
            eccp = <Txx-Tyy>/<Txx+Tyy> '''
        ed[ed<1.0E-10] = 1.0E-10
        pre = self.eos.f_P(ed)

        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)

        Tyy = (ed + pre)*u0*u0*vy*vy + pre
        Txx = (ed + pre)*u0*u0*vx*vx + pre
        T0x = (ed + pre)*u0*u0*vx
        v2 = (Txx - Tyy).sum() / (Txx + Tyy).sum()
        v1 = T0x.sum() / (Txx + Tyy).sum()
        return v1, v2

    def mean_vr(self, ed, vx, vy, vz=0.0):
        ''' <vr> = <gamma * ed * sqrt(vx*vx + vy*vy)>/<gamma*ed>
        where <> are averaged over whole transverse plane'''
        ed[ed<1.0E-10] = 1.0E-10
        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)
        vr = (u0*ed*np.sqrt(vx*vx + vy*vy)).sum() / (u0*ed).sum()
        return vr

    def total_entropy(self, tau, ed, vx, vy, vz=0.0):
        '''get the total entropy as a function of time'''
        ed[ed<1.0E-10] = 1.0E-10
        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)
        return (u0*self.eos.f_S(ed)).sum() * tau * self.cfg.DX * self.cfg.DY


    def ecc_vs_rapidity(self, bulk):
        ''' bulk = self.h_ev.reshape(NX, NY, NZ, 4)'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        ecc1 = np.empty(NZ)
        ecc2 = np.empty(NZ)
        for k in range(NZ):
            ed = bulk[:,:,k,0]
            vx = bulk[:,:,k,1]
            vy = bulk[:,:,k,2]
            vz = bulk[:,:,k,3]
            ecc1[k], ecc2[k] = self.eccp(ed, vx, vy, vz)
        return ecc1, ecc2

        
    def save(self):
        # use absolute path incase call bulkinfo.save() from other directory
        path_out = os.path.abspath(self.cfg.fPathOut)

        np.savetxt(path_out + '/avg.dat',
                   np.array(zip(self.time, self.eccp_vs_tau, self.edcent, self.entropy, self.vr)),
                   header='tau, eccp, ed(0,0,0), stotal, <vr>')

        self.f_hdf5.create_dataset('coord/tau', data = self.time)
        self.f_hdf5.create_dataset('coord/x', data = self.x)
        self.f_hdf5.create_dataset('coord/y', data = self.y)
        self.f_hdf5.create_dataset('coord/etas', data = self.z)

        self.f_hdf5.create_dataset('avg/eccp', data = np.array(self.eccp_vs_tau))
        self.f_hdf5.create_dataset('avg/edcent', data = np.array(self.edcent))
        self.f_hdf5.create_dataset('avg/Tcent', data = self.eos.f_T(np.array(self.edcent)))
        self.f_hdf5.create_dataset('avg/entropy', data = np.array(self.entropy))
        self.f_hdf5.create_dataset('avg/vr', data = np.array(self.vr))

        self.f_hdf5.close()




