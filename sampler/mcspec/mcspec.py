#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Di 05 Apr 2016 17:03:27 CEST

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import os
import pandas as pd
from time import time
import math

try:
    # used in python 2.*
    from StringIO import StringIO as fstring
except ImportError:
    # used in python 3.*
    from io import StringIO as fstring



class mcspec(object):
    def __init__(self, events_str, rapidity_kind='eta', fpath='./'):
        self.rapidity_col = 4
        if rapidity_kind == 'eta':
            self.rapidity_col = 6

        self.events = [np.genfromtxt(fstring(event)) for event
                                in events_str.split('#finished')[:-1]]

        self.num_of_events = len(self.events)
        print('in mcspec, num of events=', self.num_of_events)

        self.fpath = fpath


    def pt_differential_vn(self, n=2, pid='211', pt_min=0.3, pt_max=2.5, eta_max=2.0):
        pts = np.linspace(0.3, 2.5, 21)

        avg2_list = np.zeros(self.num_of_events)
        avg4_list = np.zeros(self.num_of_events)
        avg2_prime_list = np.zeros((self.num_of_events, len(pts)))
        avg4_prime_list = np.zeros((self.num_of_events, len(pts)))

        for idx, spec in enumerate(self.events):
            self.rapidity = spec[:, self.rapidity_col]
            # pion, kaon, proton, ... pid
            self.pid = spec[:, 5]

            self.px = spec[:, 1]
            self.py = spec[:, 2]
            self.pt = np.sqrt(self.px*self.px + self.py*self.py)
            self.phi = np.arctan2(self.py, self.px)

            # use charged particle for reference flow
            avg2_list[idx], avg4_list[idx] = self.avg(n, pid, pt1=0, pt2=4.2,
                                                  rapidity1=-5.5, rapidity2=5.5)

            for ipt, pt in enumerate(pts):
                avg2_prime_list[idx, ipt], avg4_prime_list[idx, ipt] = self.avg_prime(
                        n, pid, pt_min=pt-0.2, pt_max=pt+0.2, eta_min=-eta_max, eta_max=eta_max)

        vn2, vn4 = self.differential_flow(avg2_list, avg4_list,
                                               avg2_prime_list, avg4_prime_list)

        return pts, vn2, vn4

    def vn_vs_eta(self, pid='charged', make_plot=False):
        # <<2>> as a function of eta
        avg2_vs_eta = np.zeros(20)

        eta = np.linspace(-5, 5, 20)
        avg22_prime_list = np.zeros((self.num_of_events, 20))
        avg32_prime_list = np.zeros((self.num_of_events, 20))
        avg42_prime_list = np.zeros((self.num_of_events, 20))

        avg24_prime_list = np.zeros((self.num_of_events, 20))
        avg34_prime_list = np.zeros((self.num_of_events, 20))
        avg44_prime_list = np.zeros((self.num_of_events, 20))

        avg22_ref_list = np.zeros(self.num_of_events)
        avg32_ref_list = np.zeros(self.num_of_events)
        avg42_ref_list = np.zeros(self.num_of_events)

        avg24_ref_list = np.zeros(self.num_of_events)
        avg34_ref_list = np.zeros(self.num_of_events)
        avg44_ref_list = np.zeros(self.num_of_events)

        for idx, spec in enumerate(self.events):
            self.rapidity = spec[:, self.rapidity_col]
            # pion, kaon, proton, ... pid
            self.pid = spec[:, 5]

            self.px = spec[:, 1]
            self.py = spec[:, 2]
            self.pt = np.sqrt(self.px*self.px + self.py*self.py)
            self.phi = np.arctan2(self.py, self.px)

            # The reference particles
            avg22_ref_list[idx], avg24_ref_list[idx] = self.avg(2, pid,
                    pt1=0, pt2=5.0, rapidity1=-0.8, rapidity2=0.8)
            avg32_ref_list[idx], avg34_ref_list[idx] = self.avg(3, pid,
                    pt1=0, pt2=5.0, rapidity1=-0.8, rapidity2=0.8)
            avg42_ref_list[idx], avg44_ref_list[idx] = self.avg(4, pid,
                    pt1=0, pt2=5.0, rapidity1=-0.8, rapidity2=0.8)

            for ih in range(20):
                hmin = eta[ih] - 0.25
                hmax = eta[ih] + 0.25
                avg22_prime_list[idx, ih], avg24_prime_list[idx, ih] = self.avg_prime(2, pid,
                        pt_min=0, pt_max=5.0, eta_min=hmin, eta_max=hmax)
                avg32_prime_list[idx, ih], avg34_prime_list[idx, ih] = self.avg_prime(3, pid,
                        pt_min=0, pt_max=5.0, eta_min=hmin, eta_max=hmax)
                avg42_prime_list[idx, ih], avg44_prime_list[idx, ih] = self.avg_prime(4, pid,
                        pt_min=0, pt_max=5.0, eta_min=hmin, eta_max=hmax)

        # ignore the NAN in mean calculation if there is no particles in one rapidity bin in a event
        v22, v24 = self.differential_flow(avg22_ref_list, avg24_ref_list, avg22_prime_list, avg24_prime_list)
        v32, v34 = self.differential_flow(avg32_ref_list, avg34_ref_list, avg32_prime_list, avg34_prime_list)
        v42, v44 = self.differential_flow(avg42_ref_list, avg44_ref_list, avg42_prime_list, avg44_prime_list)

        np.savetxt(os.path.join(self.fpath, 'vn24_vs_eta.txt'), zip(eta, v22, v32, v42, v24, v34, v44))

        if make_plot:
            plt.plot(eta, v22, label='v2{2}')
            plt.plot(eta, v32, label='v3{2}')
            plt.plot(eta, v42, label='v4{2}')

            plt.legend(loc='best')
            plt.show()



    def qn(self, n, pid='211', pt1=0.0, pt2=3.0, rapidity1=-1, rapidity2=1):
        '''return the Qn cumulant vector for particles with pid in
        pt range [pt1, pt2] and rapidity range [rapidity1, rapidity2]
        Params:
            :param n: int, order of the Qn cumulant vector
            :param pid: string, 'charged', '211', '321', '2212' for
                charged particles, pion+, kaon+ and proton respectively
            :param pt1: float, lower boundary for transverse momentum
            :param pt2: float, upper boundary for transverse momentum
            :param rapidity1: float, lower boundary for rapidity
            :param rapidity2: float, upper boundary for rapidity
        Return:
            particle_of_interest, multiplicity, Qn = sum_i^m exp(i n phi) '''

        # poi stands for particle of interest
        particle_of_interest = None

        def multi_and(*args):
            '''select elements of one numpy array that satisfing multiple situations'''
            selected = np.ones_like(args[0], dtype=np.bool)
            for array_i in args:
                selected = np.logical_and(selected, array_i)
            return selected

        if pid == 'charged':
            particle_of_interest = multi_and(self.pt>pt1, self.pt<pt2,
                                             self.rapidity > rapidity1,
                                             self.rapidity < rapidity2)
        else:
            particle_of_interest = multi_and(self.pid == int(pid), self.pt>pt1,
                                             self.pt<pt2, self.rapidity > rapidity1,
                                             self.rapidity < rapidity2)

        multiplicity = np.count_nonzero(particle_of_interest)

        return  particle_of_interest, multiplicity, np.exp(1j*n*self.phi[particle_of_interest]).sum()


    def avg(self, n, pid='211', pt1=0.0, pt2=3.0, rapidity1=-1, rapidity2=1):
        '''return the 2- and 4- particle cumulants for particles with pid in
        pt range [pt1, pt2] and rapidity range [rapidity1, rapidity2]
        Params:
            :param n: int, order of the Qn cumulant vector
            :param pid: string, 'charged', '211', '321', '2212' for
                charged particles, pion+, kaon+ and proton respectively
            :param pt1: float, lower boundary for transverse momentum
            :param pt2: float, upper boundary for transverse momentum
            :param rapidity1: float, lower boundary for rapidity
            :param rapidity2: float, upper boundary for rapidity
        Return:
            cn{2} = <<2>>
            cn{4} = <<4>> - 2<<2>>**2 '''
        POI_0, M, Qn = self.qn(n, pid, pt1, pt2, rapidity1, rapidity2)
        Qn_square = Qn * Qn.conjugate()
        avg2 = ((Qn_square - M)/float(M*(M-1))).real

        POI_1, M2, Q2n = self.qn(2*n, pid, pt1, pt2, rapidity1, rapidity2)
        Q2n_square = Q2n * Q2n.conjugate()

        term1 = (Qn_square**2 + Q2n_square - 2*(Q2n*Qn.conjugate()**2).real
                )/float(M*(M-1)*(M-2)*(M-3))
        term2 = 2*(2*(M-2)*Qn_square - M*(M-3))/float(M*(M-1)*(M-2)*(M-3))
        avg4 = term1 - term2
        return avg2.real, avg4.real


    def avg_prime(self, n, pid, pt_min, pt_max, eta_min, eta_max,
                  eta_ref_min = -0.8, eta_ref_max = 0.8,
                  pt_ref_min = 0.0, pt_ref_max = 5.0):
        '''return <2'> and <4'> for particles with pid in the 
        range ( pt_min < pt < pt_max ) and ( eta_min < eta < eta_max )'''
        # Qn from reference flow particles
        REF_0, M, Qn = self.qn(n, pid, pt_ref_min, pt_ref_max, eta_ref_min, eta_ref_max)
        REF_1, M2, Q2n = self.qn(2*n, pid, pt_ref_min, pt_ref_max, eta_ref_min, eta_ref_max)

        # particle of interest
        POI_0, mp, pn = self.qn(n, pid, pt_min, pt_max, eta_min, eta_max)
 
        # mq, qn: labeled as both POI and REF
        POI_AND_REF = REF_0 & POI_0
        mq = np.count_nonzero(POI_AND_REF)
        qn = np.exp(1j*n*self.phi[POI_AND_REF]).sum()

        POI_1, mq2, q2n = self.qn(2*n, pid, pt_min, pt_max, eta_min, eta_max)

        avg2_prime = (pn * Qn.conjugate() - mq)/(mp * M - mq)
        avg4_prime = (pn * Qn * Qn.conjugate()**2 - q2n * Qn.conjugate()**2 - pn * Qn * Q2n.conjugate()
                - 2 * M * pn * Qn.conjugate() - 2 * mq * Qn * Qn.conjugate() +
                7 * qn * Qn.conjugate() - Qn * qn.conjugate() + q2n * Q2n.conjugate() 
                + 2 * pn * Qn.conjugate() + 2 * mq * M - 6 * mq ) / (
                        (mp * M - 3 * mq) * (M - 1) * (M - 2))

        return avg2_prime, avg4_prime


                               
    def differential_flow(self, avg2_list, avg4_list,
                                avg2_prime_list, avg4_prime_list):
        '''return the differential flow vs transverse momentum
        Params: 
            :param avg2_list: ebe <2>
            :param avg4_list: ebe <4>

        Returns:
            pt_array, vn{2} array, vn{4} array '''
        avg2 = np.nanmean(avg2_list)
        avg4 = np.nanmean(avg4_list)
        avg2_prime = np.nanmean(avg2_prime_list, axis=0)
        avg4_prime = np.nanmean(avg4_prime_list, axis=0)

        cn2 = avg2
        cn4 = avg4 - 2 * avg2 * avg2

        dn2 = avg2_prime
        dn4 = avg4_prime - 2 * avg2_prime * avg2

        vn2 = dn2 / np.sqrt(cn2)
        vn4 = - dn4 / np.power(-cn4, 0.75)

        return vn2, vn4

    def plot_vn_pt(self, n=2, make_plot=False):
        '''plot vn as a function of pt '''
        pts, vn2_pion,    vn4_pion = self.pt_differential_vn(n=n, pid='211')
        pts, vn2_kaon,    vn4_kaon = self.pt_differential_vn(n=n, pid='321')
        pts, vn2_proton,  vn4_proton = self.pt_differential_vn(n=n, pid='2212')
        pts, vn2_charged, vn4_charged = self.pt_differential_vn(n=n, pid='charged')

        np.savetxt(os.path.join(self.fpath, 'v%s_2_vs_pt.txt'%n), zip(pts,
                   vn2_pion, vn2_kaon, vn2_proton, vn2_charged))

        np.savetxt(os.path.join(self.fpath, 'v%s_4_vs_pt.txt'%n), zip(pts,
                   vn4_pion, vn4_kaon, vn4_proton, vn4_charged))

        print("v%s finished!"%n)

        if make_plot:
            plt.plot(pts, vn4_pion, label='v%s{2} pion'%n)
            plt.plot(pts, vn4_kaon, label='v%s{2} kaon'%n)
            plt.plot(pts, vn4_proton, label='v%s{2} proton'%n)

            plt.legend(loc='best')
            plt.show()



from subprocess import call, check_output

def calc_vn(fpath, over_sampling=1000, make_plot=False):
    cwd = os.getcwd()
    os.chdir('../build')
    #call(['cmake', '..'])
    #call(['make'])
    cmd = ['./main', fpath, 'true', 'true', '%s'%over_sampling]

    proc = check_output(cmd)

    stio = fstring()
    stio.write(proc)

    mc = mcspec(stio.getvalue(), fpath=fpath)

    mc.vn_vs_eta(make_plot = make_plot)
    mc.plot_vn_pt(n=4)
    mc.plot_vn_pt(n=3)
    mc.plot_vn_pt(n=2, make_plot=make_plot)

    os.chdir(cwd)



if __name__=='__main__':
    t1 = time()

    import sys

    fpath = '/lustre/nyx/hyihp/lpang/trento_ini/bin/pbpb2p76/20_30/n2/mean'

    if len(sys.argv) == 2:
        fpath = sys.argv[1]

    calc_vn(fpath, over_sampling=5000, make_plot=True)


