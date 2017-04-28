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


class mcspec(object):
    def __init__(self, fpath, rapidity_kind='eta'):
        fname = os.path.join(fpath, 'mc_particle_list.dat')

        #spec = np.loadtxt(fname)
        spec = pd.read_csv(fname, sep=' ', header=None, dtype=np.float32,
                           skiprows=1).values

        # the 4th column stores Y = 0.5ln((E+pz)/(E-pz))
        rapidity_col = 4

        # the 6th column stores eta = 0.5ln((p+pz)/(p-pz))
        if rapidity_kind == 'eta':
            rapidity_col = 6

        self.rapidity = spec[:, rapidity_col]

        # pion, kaon, proton, ... pid
        self.pid = spec[:, 5]

        self.px = spec[:, 1]
        self.py = spec[:, 2]
        self.pt = np.sqrt(self.px*self.px + self.py*self.py)

        self.phi = np.arctan2(self.py, self.px)

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
            Qn = sum_i^m exp(i n phi) '''

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

        return  multiplicity, np.exp(1j*n*self.phi[particle_of_interest]).sum()


    def cn(self, n, pid='211', pt1=0.0, pt2=3.0, rapidity1=-1, rapidity2=1):
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
        M, Qn = self.qn(n, pid, pt1, pt2, rapidity1, rapidity2)
        Qn_square = Qn * Qn.conjugate()
        avg2 = ((Qn_square - M)/float(M*(M-1))).real

        M2, Q2n = self.qn(2*n, pid, pt1, pt2, rapidity1, rapidity2)
        Q2n_square = Q2n * Q2n.conjugate()
        term1 = (Qn_square**2 + Q2n_square - 2*(Q2n*Qn.conjugate()**2).real
                )/float(M*(M-1)*(M-2)*(M-3))
        term2 = 2*(2*(M-2)*Qn_square - M*(M-3))/float(M*(M-1)*(M-2))
        avg4 = term1 - term2

        return avg2.real, (avg4 - 2*avg2**2).real
                                
    def differential_flow(self, n, pid='211'):
        '''return the differential flow vs transverse momentum
        Params: 
            :param n: int, harmonic order
            :param pid: string, particle type

        Returns:
            pt_array, vn{2} array, vn{4} array '''
        pts = np.linspace(0.1, 4.1, 20, endpoint=True)
        rapidity_range = [-1.0, 1.0]

        # Qn from reference flow particles
        M, Qn = self.qn(n, pid, 0, 4.2, -8, 8)
        M2, Q2n = self.qn(2*n, pid, 0, 4.2, -8, 8)

        vn_2 = np.empty_like(pts)
        vn_4 = np.empty_like(pts)

        cn2, cn4 = self.cn(n, pid, 0, 4.2, -8, 8)
        avg2 = cn2

        for i, pt in enumerate(pts):
            mp, pn = self.qn(n, pid, pt-0.2, pt+0.2, rapidity_range[0], rapidity_range[1])
            mq, qn = mp, pn
            mq2, q2n = self.qn(2*n, pid, pt-0.2, pt+0.2, rapidity_range[0], rapidity_range[1])
            avg2_prime = (pn * Qn.conjugate() - mq)/(mp * M - mq)
            avg4_prime = (pn * Qn * Qn.conjugate()**2 - q2n * Qn.conjugate()**2 - pn * Qn * Q2n.conjugate()
                    - 2 * M * pn * Qn.conjugate() - 2 * mq * Qn * Qn.conjugate() +
                    7 * qn * Qn.conjugate() - Qn * qn.conjugate() + q2n * Q2n.conjugate() 
                    + 2 * pn * Qn.conjugate() + 2 * mq * M - 6 * mq ) / (
                            (mp * M - 3 * mq) * (M - 1) * (M - 2) * (M-3))

            dn2 = avg2_prime
            dn4 = avg4_prime - 2 * avg2 * avg2_prime

            vn_2[i] = dn2.real / math.sqrt(cn2)
            vn_4[i] = - dn4.real / math.pow(-cn4, 0.75)

        return pts, vn_2, vn_4



if __name__=='__main__':
    t1 = time()
    spec = mcspec('/lustre/nyx/hyihp/lpang/auau200_results/cent20_30/etas0p08/event1/')
    t2 = time()
    print('read data spent', t2-t1, ' s') 
    pt, pion_vn2, pion_vn4 = spec.differential_flow(n=2, pid='211')

    pt, kaon_vn2, kaon_vn4 = spec.differential_flow(n=2, pid='321')

    pt, proton_vn2, proton_vn4 = spec.differential_flow(n=2, pid='2212')

    plt.plot(pt, pion_vn2, label='pion vn2')
    plt.plot(pt, pion_vn4, label='pion vn4')

    plt.plot(pt, kaon_vn2, label='kaon vn2')
    plt.plot(pt, kaon_vn4, label='kaon vn4')

    plt.plot(pt, proton_vn2, label='proton vn2')
    plt.plot(pt, proton_vn4, label='proton vn4')

    plt.legend(loc='best')
    plt.show()


