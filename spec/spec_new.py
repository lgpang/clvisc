#!/usr/bin/python
import numpy as np

import os
from const import *
import cmath

class Spec:
    '''Calc dN/dY, 1/(2pi)dN/dYptdpt, v2(pt)
            dN/deta, 1/(2pi)dN/detaptdpt 
       Args:
           path: the path for the spec
           pid:  string, the pid for the particle, for charged hadron 
                 pid='Charged', for pion+, pid='211'
           reso: default(False), after or before resonance decay
           rapidity_kind: 'Y' for rapidity and 'Eta' for pseudo-rapidity
           '''
    def __init__(self, path, pid='211', reso=False, rapidity_kind='Y'):
        self.path = path
        self.rapidity_kind = rapidity_kind
        self.pid = pid

        if pid == 'Charged':
            reso = False

        # fname for spec
        if reso == False:
            fname = self.path+"/dNd{rapidity}PtdPtdPhi_{pid}.dat".format(
                rapidity=self.rapidity_kind, pid=pid)

            self.strout = "_{pid}.dat".format(pid=pid)
        else:
            fname = self.path+"/dNd{rapidity}PtdPtdPhi_Reso{pid}.dat".format(
                rapidity=self.rapidity_kind, pid=pid)
            self.strout = "_Reso{pid}.dat".format(pid=pid)

        self.spec = np.loadtxt(fname).reshape(NY, NPT, NPHI)/(HBARC**3.0)

        self.get_dNdYPtdPt2D()
        self.get_dNdY()

    def get_ptspec(self, ylo=-0.8, yhi=0.8, comment=""):
        self.get_dNPtdPt_over2pi(ylo, yhi, comment)


    def get_vn(self, ylo=-2.5, yhi=2.5, event_plane_window=(3.3, 4.8), comment=""):
        ''' calculate the pt differential vn in event plane method
        Args:
           ylo, yhi: calc vn for particles in the rapidity window [ylo, yhi]
           event_plane_window: Tupe (ylo_ep, yhi_ep) for event plane calculation
           comment: additional information to add to the output file name

        Return:
           The pt, vn(pt) is saved in text file vn_*.dat
        '''
        self.get_event_planes(ylo=event_plane_window[0], 
                              yhi=event_plane_window[1])
        print('psi[23456]=', self.event_plane[1:6])
        self.get_Vn_vs_pt(ylo, yhi, comment)
        pass


    def get_dNdYPtdPt2D(self):
        '''Integrate over phi to get dN/dYptdpt
        for spec stored in fname'''
        self.dNdYPtdPt2D = np.zeros( (NY, NPT) ) 
        
        for i in range(NY):
            for j in range(NPT):
                spec_along_phi = self.spec[i,j,:]
                self.dNdYPtdPt2D[i,j] = phi_integral(spec_along_phi)

    ######################################################################
    def get_dNdY(self):
        ''' Get dNdY or dNdEta for spec with pid
        arg:
            pid (string, default='211') 
            '211' for pion+, '321' for Kaon+, 2212 for proton, 'Charged' for 

            reso (bool, default=False) with or without resonance decay
            '''
        # print self.specs
        dNdY = np.zeros(NY)
        for i in range(NY):
            spec_along_pt = self.dNdYPtdPt2D[i,:]
            dNdY[i] = pt_integral(PT*spec_along_pt)

        fout_name = self.path + "/dNd{rapidity}".format( \
            rapidity = self.rapidity_kind) + self.strout

        np.savetxt(fout_name, np.array(list(zip(Y, dNdY))), \
            header='#rapidity dN/d{rapidity}'.format( \
            rapidity=self.rapidity_kind))

    ######################################################################
    # @profile
    def get_dNPtdPt_over2pi(self, ylo=-1.3, yhi=1.3, comment=""):
        ''' return (1/2pi)dN/(dYPtdPt) and (1/2pi)dN/(dEtaPtdPt) '''
        # print self.specs
        dNPtdPt_over2pi = np.zeros( NPT )
        for j in range(NPT):
            spec_along_y = self.dNdYPtdPt2D[:,j]
            dNPtdPt_over2pi[j] = rapidity_integral(spec_along_y, ylo, yhi)
        
        dNPtdPt_over2pi /= ( 2.0*np.pi*(yhi-ylo) )

        fout_name = self.path + "/dNd{rapidity}PtdPt_over_2pi{comment}".format(
                rapidity=self.rapidity_kind, comment=comment) + self.strout

        np.savetxt(fout_name, np.array(list(zip(PT, dNPtdPt_over2pi))),
           header='#PT (1/2pi)dN/d{rapidity}ptdpt'.format(
           rapidity = self.rapidity_kind))
    
    ######################################################################
    def get_dNdPtdPhi(self, ylo=-0.5, yhi=0.5):
        ''' return dN/(dPtdPhi) for particles in rapidity range [ylo, yhi]'''
        dNdPtdPhi2D = np.zeros((NPT, NPHI))
        for i in range(NPT):
            for j in range(NPHI):
                spec_along_y = self.spec[:,i,j]
                dNdPtdPhi2D[i,j] = rapidity_integral(spec_along_y,
                        ylo, yhi) * PT[i]
        return dNdPtdPhi2D

    def get_event_planes(self, ylo=-0.5, yhi=0.5):
        ''' return psi_1, psi_2, psi_3, psi_4, psi_5, psi_6 for all the 
        particles (all pt range) in rapidity window [ylo, yhi]'''
        spec_along_phi = np.zeros(NPHI)
        dNdPtdPhi2D = self.get_dNdPtdPhi(ylo, yhi)
        for j in range(NPHI):
            spec_along_pt = dNdPtdPhi2D[:,j]
            spec_along_phi[j] = pt_integral(spec_along_pt)

        Vn = np.zeros(7)
        self.event_plane = np.zeros(7)
        norm = phi_integral(spec_along_phi)
        for n in xrange(1, 7):
            Vn[n], self.event_plane[n] = cmath.polar(phi_integral(
                        spec_along_phi*np.exp(1j*n*PHI))/norm)
            self.event_plane[n] /= float(n)
            #self.event_plane[n] = 0.0



    def get_Vn_vs_pt(self, ylo=-0.5, yhi=0.5, comment=''):
        ''' return pt differential Vn=vn*exp(i n Psi_n) '''
        Vn_vs_pt = np.zeros(shape=(7, NPT))
        angles = np.zeros(shape=(7, NPT))

        dNdPtdPhi2D = self.get_dNdPtdPhi(ylo, yhi)
        for i in range(NPT):
            spec_along_phi = dNdPtdPhi2D[i,:]
            norm_factor = phi_integral(spec_along_phi)
            for n in xrange(1, 7):
                Vn_vs_pt[n, i], angles[n, i] = cmath.polar(phi_integral(
                    spec_along_phi*np.exp(1j*n*(PHI-self.event_plane[n])))
                    /norm_factor)

        fout_name = self.path + "/vn%s"%comment + self.strout

        np.savetxt(fout_name, np.array(list(zip(PT, Vn_vs_pt[1,:],
                                               Vn_vs_pt[2,:],
                                               Vn_vs_pt[3,:],
                                               Vn_vs_pt[4,:],
                                               Vn_vs_pt[5,:],
                                               Vn_vs_pt[6,:],))),
                   header='#PT v1 v2 v3 v4 v5 v6')
 
