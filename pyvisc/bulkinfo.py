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
#import matplotlib.pyplot as plt

import logging

logging.basicConfig(filename='bulkinfo.log', level=logging.DEBUG)


class BulkInfo(object):
    '''The bulk information like:
       ed(x), ed(y), ed(eta), T(x), T(y), T(eta)
       vx, vy, veta, ecc_x, ecc_p'''
    def __init__(self, cfg, ctx, queue, eos_table, compile_options):
        self.cfg = cfg
        self.ctx = ctx
        self.queue = queue
        self.eos_table = eos_table
        self.compile_options = list(compile_options)

        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ
        self.h_ev = np.zeros((NX*NY*NZ, 4), cfg.real)
        self.h_pi = np.zeros(10*NX*NY*NZ, self.cfg.real)

        # one dimensional
        self.ex, self.ey, self.ez = [], [], []
        self.vx, self.vy, self.vz = [], [], []

        # in transverse plane (z==0)
        self.exy, self.vx_xy, self.vy_xy, self.vz_xy = [], [], [], []

        self.pixx_xy, self.piyy_xy, self.pitx_xy = [], [], []

        # in reaction plane
        self.exz, self.vx_xz, self.vy_xz, self.vz_xz = [], [], [], []

        self.ecc2_vs_rapidity = []
        self.ecc1_vs_rapidity = []
        self.time = []
        self.edmax = []
        self.__loadAndBuildCLPrg()
        self.eos = Eos(cfg.IEOS)

        self.x = np.linspace(-floor(NX/2)*cfg.DX, floor(NX/2)*cfg.DX, NX, endpoint=True)
        self.y = np.linspace(-floor(NY/2)*cfg.DY, floor(NY/2)*cfg.DY, NY, endpoint=True)
        self.z = np.linspace(-floor(NZ/2)*cfg.DZ, floor(NZ/2)*cfg.DZ, NZ, endpoint=True)



    def __loadAndBuildCLPrg(self):
        #load and build *.cl programs with compile self.compile_options
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

            __kernel void get_pimn(__global real * d_pi,
                                 __global real * d_pixx_xy,
                                 __global real * d_piyy_xy,
                                 __global real * d_pitx_xy)
            {
                int gid_x = get_global_id(0);
                int gid_y = get_global_id(1);

                int oid = gid_x*NY*(NZ/2) + gid_y*(NZ/2) + NZ/2;

                int nid = gid_x*NY + gid_y;

                d_pixx_xy[nid] = d_pi[10*oid + 4];
                d_piyy_xy[nid] = d_pi[10*oid + 7];
                d_pitx_xy[nid] = d_pi[10*oid + 1];
            }
            '''
        self.kernel_edslice = cl.Program(self.ctx, edslice_src).build(
                                         options=self.compile_options)

    def get(self, tau, d_ev1, edmax, d_pi=None):
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
                d_evx0, d_evy0, d_evz0, d_evxy, d_evxz, d_evyz).wait()

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

        logging.debug('d_pi is not None: %s'%(d_pi is not None))
        if d_pi is not None:
            h_pixx = np.zeros(NX*NY, self.cfg.real)
            h_piyy = np.zeros(NX*NY, self.cfg.real)
            h_pitx = np.zeros(NX*NY, self.cfg.real)
            d_pixx = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_pixx.nbytes)
            d_piyy = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_pixx.nbytes)
            d_pitx = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_pixx.nbytes)
            self.kernel_edslice.get_pimn(self.queue, (NX, NY), None, d_pi,
                    d_pixx, d_piyy, d_pitx).wait()

            cl.enqueue_copy(self.queue, h_pixx, d_pixx).wait()
            self.pixx_xy.append(h_pixx.reshape(NX, NY))

            cl.enqueue_copy(self.queue, h_piyy, d_piyy).wait()
            self.piyy_xy.append(h_piyy.reshape(NX, NY))

            cl.enqueue_copy(self.queue, h_pitx, d_pitx).wait()
            self.pitx_xy.append(h_pitx.reshape(NX, NY))


    def eccp(self, ed, vx, vy, vz=0.0, pixx=None, piyy=None, pitx=None):
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

        v2 = 0.0

        if pixx is not None:
            pi_sum = (pixx + piyy).sum()
            pi_dif = (pixx - piyy).sum()
            v2 = ((Txx - Tyy).sum() + pi_dif) / ((Txx + Tyy).sum() + pi_sum)
        else:
            v2 = (Txx - Tyy).sum()/(Txx + Tyy).sum()

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
        
    def save(self, viscous_on=False):
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

        entropy = []
        vr = []
        ecc2 = []
        ecc1 = []
        ecc2_visc = []
        for idx, exy in enumerate(self.exy):
            vx= self.vx_xy[idx]
            vy= self.vy_xy[idx]
            np.savetxt(path_out+'/edxy%d.dat'%idx, exy)
            np.savetxt(path_out+'/Txy%d.dat'%idx, self.eos.f_T(exy))
            np.savetxt(path_out+'/vx_xy%d.dat'%idx, vx)
            np.savetxt(path_out+'/vy_xy%d.dat'%idx, vy)
            tmp0, tmp1 = self.eccp(exy, vx, vy)
            ecc1.append(tmp0)
            ecc2.append(tmp1)
            vr.append(self.mean_vr(exy, vx, vy))
            tau = self.time[idx]
            entropy.append(self.total_entropy(tau, exy, vx, vy))

            if viscous_on:
                pixx = self.pixx_xy[idx]
                piyy = self.piyy_xy[idx]
                pitx = self.pitx_xy[idx]
                ecc_visc1, ecc_visc2 = self.eccp(exy, vx, vy,
                        pixx=pixx, piyy=piyy, pitx=pitx)

                ecc2_visc.append(ecc_visc2)

        
        for idx, exz in enumerate(self.exz):
            np.savetxt(path_out+'/ed_xz%d.dat'%idx, exz)
            np.savetxt(path_out+'/vx_xz%d.dat'%idx, self.vx_xz[idx])
            np.savetxt(path_out+'/vy_xz%d.dat'%idx, self.vy_xz[idx])
            np.savetxt(path_out+'/vz_xz%d.dat'%idx, self.vz_xz[idx])
            np.savetxt(path_out+'/T_xz%d.dat'%idx, self.eos.f_T(exz))

        np.savetxt(path_out + '/eccp.dat',
                   np.array(zip(self.time, ecc2)), header='tau  eccp')

        if viscous_on:
            np.savetxt(path_out + '/eccp_visc.dat',
                   np.array(zip(self.time, ecc2_visc)), header='tau  eccp_visc')


        np.savetxt(path_out + '/Tmax.dat',
                   np.array(zip(self.time, self.eos.f_T(self.edmax))),
                   header='tau, Tmax')

        np.savetxt(path_out + '/edmax.dat',
                   np.array(zip(self.time, self.edmax)),
                   header='tau, edmax')

        np.savetxt(path_out + '/vr.dat',
                   np.array(zip(self.time, vr)), header='tau <vr>')

        np.savetxt(path_out + '/entropy.dat',
                   np.array(zip(self.time, entropy)), header='tau  entropy')
