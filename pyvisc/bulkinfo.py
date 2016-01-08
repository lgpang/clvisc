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

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
from eos.eos import Eos
#import matplotlib.pyplot as plt


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
        self.h_ev = np.zeros((NX*NY*NZ, 4), cfg.real)

        # one dimensional
        self.ex, self.ey, self.ez = [], [], []
        self.vx, self.vy, self.vz = [], [], []

        # in transverse plane (z==0)
        self.exy, self.vx_xy, self.vy_xy, self.vz_xy = [], [], [], []

        # in reaction plane
        self.exz, self.vx_xz, self.vy_xz, self.vz_xz = [], [], [], []

        self.ecc_x = []
        self.ecc_p = []
        self.ecc2_vs_rapidity = []
        self.ecc1_vs_rapidity = []
        self.time = []
        self.edmax = []
        self.__loadAndBuildCLPrg()
        self.eos = Eos(cfg.IEOS)

        self.x = np.linspace(-NX/2*cfg.DX, NX/2*cfg.DX, NX)
        self.y = np.linspace(-NY/2*cfg.DY, NY/2*cfg.DY, NY)
        self.z = np.linspace(-NZ/2*cfg.DZ, NZ/2*cfg.DZ, NZ)



    def __loadAndBuildCLPrg(self):
        #load and build *.cl programs with compile self.gpu_defines
        edslice_src = '''#include"real_type.h"
            __kernel void get_ed(__global real4 * d_ev,
                                 __global real4 * d_ev_x0,
                                 __global real4 * d_ev_y0,
                                 __global real4 * d_ev_z0,
                                 __global real4 * d_ev_xy,
                                 __global real4 * d_ev_xz,
                                 __global real4 * d_ev_yz) {
                int gid = get_global_id(0);
                if ( gid < NX ) {
                    int j = NY/2; 
                    int k = NZ/2;
                    d_ev_x0[gid] = d_ev[gid*NY*NZ + j*NZ + k];

                    int i = gid;
                    for ( j = 0; j< NY; j ++ ) {
                        d_ev_xy[i*NY+j] = d_ev[i*NY*NZ + j*NZ + k];
                    }

                    j = NY/2;
                    for ( k = 0; k < NZ; k ++ ) {
                        d_ev_xz[i*NZ+k] = d_ev[i*NY*NZ + j*NZ + k];
                    }
                }

                if ( gid < NY ) {
                    int i = NX/2; 
                    int k = NZ/2;
                    d_ev_y0[gid] = d_ev[i*NY*NZ + gid*NZ + k];
                    int j = gid;
                    for ( k = 0; k < NZ; k ++ ) {
                        d_ev_yz[j*NZ+k] = d_ev[i*NY*NZ + j*NZ + k];
                    }
                }

                if ( gid < NZ ) {
                    int i = NX/2; 
                    int j = NY/2;
                    d_ev_z0[gid] = d_ev[i*NY*NZ + j*NZ + gid];
                }
            }
            '''
        self.kernel_edslice = cl.Program(self.ctx, edslice_src).build(
                                         options=self.compile_options)
    def get(self, tau, d_ev1, edmax):
        self.time.append(tau)
        self.edmax.append(edmax)
        mf = cl.mem_flags
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ

        self.ecc_vs_rapidity(d_ev1)

        h_ev1d = np.zeros((2000, 4), self.cfg.real)
        h_evxy = np.zeros((NX*NY, 4), self.cfg.real)
        h_evxz = np.zeros((NX*NZ, 4), self.cfg.real)
        h_evyz = np.zeros((NY*NZ, 4), self.cfg.real)

        d_evx0 = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_ev1d.nbytes)
        d_evy0 = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_ev1d.nbytes)
        d_evz0 = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_ev1d.nbytes)

        d_evxy = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_evxy.nbytes)
        d_evxz = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_evxz.nbytes)
        d_evyz = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_evyz.nbytes)

        self.kernel_edslice.get_ed(self.queue, (2000,), None, d_ev1,
                d_evx0, d_evy0, d_evz0, d_evxy, d_evxz, d_evyz)

        h_evx0 = np.zeros((NX, 4), self.cfg.real)
        h_evy0 = np.zeros((NY, 4), self.cfg.real)
        h_evz0 = np.zeros((NZ, 4), self.cfg.real)
        cl.enqueue_copy(self.queue, h_evx0, d_evx0).wait()
        cl.enqueue_copy(self.queue, h_evy0, d_evy0).wait()
        cl.enqueue_copy(self.queue, h_evz0, d_evz0).wait()

        self.ex.append(h_evx0[:,0])
        self.ey.append(h_evy0[:,0])
        self.ez.append(h_evz0[:,0])

        self.vx.append(h_evx0[:,1])
        self.vy.append(h_evy0[:,2])
        self.vz.append(h_evz0[:,3])

        cl.enqueue_copy(self.queue, h_evxy, d_evxy).wait()
        cl.enqueue_copy(self.queue, h_evxz, d_evxz).wait()
        cl.enqueue_copy(self.queue, h_evyz, d_evyz).wait()

        self.exy.append(h_evxy[:,0].reshape(NX, NY))
        self.vx_xy.append(h_evxy[:,1].reshape(NX, NY))
        self.vy_xy.append(h_evxy[:,2].reshape(NX, NY))

        self.exz.append(h_evxz[:,0].reshape(NX, NZ))
        self.vx_xz.append(h_evxz[:,1].reshape(NX, NZ))
        self.vy_xz.append(h_evxz[:,2].reshape(NX, NZ))
        self.vz_xz.append(h_evxz[:,3].reshape(NX, NZ))

    def eccp(self, ed, vx, vy, vz=0.0):
        ''' eccx = <y*y-x*x>/<y*y+x*x> where <> are averaged 
            eccp = <Txx-Tyy>/<Txx+Tyy> '''
        ed[ed<1.0E-10] = 1.0E-10
        pre = self.eos.f_P(ed)

        u0 = 1.0/np.sqrt(1.0 - vx*vx - vy*vy - vz*vz)
        Tyy = (ed + pre)*u0*u0*vy*vy + pre
        Txx = (ed + pre)*u0*u0*vx*vx + pre
        T0x = (ed + pre)*u0*u0*vx
        v2 = (Txx - Tyy).sum() / (Txx + Tyy).sum()
        v1 = T0x.sum() / (Txx + Tyy).sum()
        return v1, v2


    def ecc_vs_rapidity(self, d_ev):
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        cl.enqueue_copy(self.queue, self.h_ev, d_ev).wait()
        bulk = self.h_ev.reshape(NX, NY, NZ, 4)
        ecc1 = np.empty(NZ)
        ecc2 = np.empty(NZ)
        for k in range(NZ):
            ed = bulk[:,:,k,0]
            vx = bulk[:,:,k,1]
            vy = bulk[:,:,k,2]
            vz = bulk[:,:,k,3]
            ecc1[k], ecc2[k] = self.eccp(ed, vx, vy, vz)
        self.ecc1_vs_rapidity.append(ecc1)
        self.ecc2_vs_rapidity.append(ecc2)
        
    def save(self):
        # use absolute path incase call bulkinfo.save() from other directory
        path_out = os.path.abspath(self.cfg.fPathOut)

        np.savetxt(path_out+'/ex.dat', np.array(self.ex).T)
        np.savetxt(path_out+'/ey.dat', np.array(self.ey).T)
        np.savetxt(path_out+'/ez.dat', np.array(self.ez).T)

        np.savetxt(path_out+'/Tx.dat', self.eos.f_T(self.ex).T)
        np.savetxt(path_out+'/Ty.dat', self.eos.f_T(self.ey).T)
        np.savetxt(path_out+'/Tz.dat', self.eos.f_T(self.ez).T)

        np.savetxt(path_out+'/vx.dat', np.array(self.vx).T)
        np.savetxt(path_out+'/vy.dat', np.array(self.vy).T)
        np.savetxt(path_out+'/vz.dat', np.array(self.vz).T)

        if len(self.ecc2_vs_rapidity) != 0:
            np.savetxt(path_out+'/ecc2.dat',
                       np.array(self.ecc2_vs_rapidity).T)
            np.savetxt(path_out+'/ecc1.dat',
                       np.array(self.ecc1_vs_rapidity).T)

        ecc2 = []
        ecc1 = []
        for idx, exy in enumerate(self.exy):
            vx= self.vx_xy[idx]
            vy= self.vy_xy[idx]
            np.savetxt(path_out+'/edxy%d.dat'%idx, exy)
            np.savetxt(path_out+'/Txy%d.dat'%idx, self.eos.f_T(exy))
            np.savetxt(path_out+'/vx_xy%d.dat'%idx, vx)
            np.savetxt(path_out+'/vy_xy%d.dat'%idx, vy)

            ecc1.append(self.eccp(exy, vx, vy)[0])
            ecc2.append(self.eccp(exy, vx, vy)[1])
        
        for idx, exz in enumerate(self.exz):
            np.savetxt(path_out+'/ed_xz%d.dat'%idx, exz)
            np.savetxt(path_out+'/vx_xz%d.dat'%idx, self.vx_xz[idx])
            np.savetxt(path_out+'/vy_xz%d.dat'%idx, self.vy_xz[idx])
            np.savetxt(path_out+'/vz_xz%d.dat'%idx, self.vz_xz[idx])
            np.savetxt(path_out+'/T_xz%d.dat'%idx, self.eos.f_T(exz))

        np.savetxt(path_out + '/eccp.dat',
                   np.array(zip(self.time, ecc2)))

        np.savetxt(path_out + '/Tmax.dat',
                   np.array(zip(self.time, self.eos.f_T(self.edmax))))
