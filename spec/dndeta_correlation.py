#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 15 Sep 2015 13:55:52 CEST

import matplotlib.pyplot as plt
import numpy as np
import os
import math

from scipy.interpolate import interp1d, splrep, splint
import spec_new as spec
import const
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from scipy.special import legendre as Pn

num_of_points = 30

def pt_integral(spec_along_pt, pt_low=0.5, pt_high=4.0):
    '''1D integration along transverse momentum in range 
    [pt_low, pt_high] '''
    tck = splrep(const.PT, spec_along_pt)
    return splint(pt_low, pt_high, tck)

def get_dndeta(event_path, ptcut=0.5):
    '''Get the charged dNdEta for pt>ptcut'''
    charged = spec.Spec(event_path, pid='211', reso=False, rapidity_kind='Eta')
    spec2d = charged.dNdYPtdPt2D
    dNdY_ptcut = []
    for k, rapidity in enumerate(const.Y):
        spec_along_pt = spec2d[k]*const.PT
        dNdY_ptcut.append(pt_integral(spec_along_pt, pt_low=ptcut))

    return dNdY_ptcut

def get_neta_ebe(cent_path, nevent=100, ptcut=0.5, cent='0_5', update_data=False):
    '''Get dndeta for all the event, 
    return np.array() with each line stores dndeta for one event'''
    saved_data = 'dndeta_%s.dat'%cent
    if os.path.exists(saved_data) and not update_data:
        neta = np.loadtxt(saved_data)
    else:
        neta = []
        for i in range(nevent):
            event_path = cent_path + '/event%d'%i
            dndeta_with_ptcut = get_dndeta(event_path, ptcut)
            neta.append(dndeta_with_ptcut)
            print 'event', i, ' finished'
        np.savetxt(saved_data, np.array(neta))

    intp_f = []
    for i in range(nevent):
        dndeta = interp1d(const.Y, neta[i], kind='cubic')
        intp_f.append(dndeta)
    return intp_f


def C(eta1, eta2, N):
    '''Cacl C(eta1, eta2) = <N_eta1*N_eta2>/<N_eta1><N_eta2>
    Args: 
        eta1, eta2:  rapidity position
        N:  dndeta(eta) as a function of eta for many events'''
    nevents = len(N)
    sum_N12 = 0.0
    sum_N1 = 0.0
    sum_N2 = 0.0
    for n in N:
        N1, N2 = n(eta1), n(eta2)
        sum_N12 += N1*N2
        sum_N1  += N1
        sum_N2  += N2
    return nevents*sum_N12/(sum_N1*sum_N2)


def Cp(eta1, N, Y=2.4):
    '''Calc Cp(eta1) with eta2 integrated over [-Y, Y]
    Cp(eta1) = int_{-Y}^{Y} C(eta1, eta2) deta2 '''
    eta = np.linspace(-Y, Y, num_of_points, endpoint=True)
    tck = splrep(eta, C(eta1, eta, N))
    return splint(-Y, Y, tck)/(2.0*Y)
    

def Cn(eta1, eta2, N, Y=2.4):
    return C(eta1, eta2, N)/(Cp(eta1, N, Y=2.4)*Cp(eta2, N, Y=2.4))


def create_cn_table(cent_dir='/scratch/hyihp/pang/ini/PbPb_Ini_b0_5_sig0p6', 
		cent='0_5', num_of_events=100, Y=2.4, update_data=False):
    fout_name='cn_b%s.dat'%cent
    # load from file if data exists
    if os.path.exists(fout_name) and not update_data:
        return np.loadtxt(fout_name)

    event = get_neta_ebe(cent_dir, nevent=num_of_events,
                         cent=cent, update_data=True)
    eta1 = np.linspace(-Y, Y, num_of_points, endpoint=True)
    eta2 = np.linspace(-Y, Y, num_of_points, endpoint=True)
    cn_table = np.empty((num_of_points, num_of_points))
    
    for i, a in enumerate(eta1):
        for j, b in enumerate(eta2):
            cn_table[i,j] = Cn(a, b, event)
            print 'i, j=', i, j, 'finished'

    # renormalize cn_table to make its average = 1.0
    cn_intp = RectBivariateSpline(eta1, eta2, cn_table)
    cn_tot = cn_intp.integral(-Y, Y, -Y, Y)
    area = 4.0*Y*Y
    cn_table = cn_table/(cn_tot/area)
    np.savetxt(fout_name, cn_table)
    return np.array(cn_table)


def T(n, eta, Y=2.4):
    '''Orthogonal polynomials for nth order'''
    return math.sqrt(n+0.5)*Pn(n)(eta/Y)
    

def get_anm(n, m, cn_table, Y=2.4):
    '''Cacl a_{n,m}=\int Cn'(eta1, eta2)*(Tn(eta1)*Tm(eta2)+Tn(eta2)Tm(eta1))/2 deta1 deta2
       Cn'(eta1, eta2) = cn_table / (cn_total/area) 
       where cn_total=\int cn_table(eta1, eta2) deta1 deta2, and area=2Y*2Y '''

    eta1 = np.linspace(-Y, Y, num_of_points, endpoint=True)
    eta2 = np.linspace(-Y, Y, num_of_points, endpoint=True)

    CT = np.empty((num_of_points, num_of_points))
    for i, a in enumerate(eta1):
        for j, b in enumerate(eta2):
            CT[i, j] = cn_table[i, j]*0.5*(T(n,a)*T(m,b)+T(n,b)*T(m,a))

    # 2d spline interpolation and integration for CT
    #    CT=Cn*0.5*(Tn1*Tm2+Tn2*Tm1)
    ct_intp = RectBivariateSpline(eta1, eta2, CT)
    return ct_intp.integral(-Y, Y, -Y, Y)
   

def test_Tn():
    eta = np.linspace(-2.4, 2.4, num_of_points)
    tn = T(6, eta)
    plt.plot(eta, tn)
    plt.show()

if __name__ == '__main__':
    Ymax = 2.4
    eta = np.linspace(-Ymax, Ymax, num_of_points, endpoint=True)
    cent_dir = '/scratch/hyihp/pang/ini/PbPb_Ini_b30_40_sig0p6'
    cent = '30_40'

    update_data = True

    cn_table = create_cn_table(cent_dir, cent,
                    num_of_events=100, update_data=update_data)
    #plt.imshow(cn_table, extent=[-2.4, 2.4, -2.4, 2.4])
    #plt.colorbar()
    #plt.show()

    fname = 'anm_%s.dat'%cent
    if os.path.exists(fname) and not update_data:
        qnn_sqrt = np.loadtxt(fname)
    else:
        ann_sqrt = np.array([
            np.sqrt(get_anm(1, 1, cn_table, Y=2.4)),
            np.sqrt(get_anm(2, 2, cn_table, Y=2.4)),
            np.sqrt(get_anm(3, 3, cn_table, Y=2.4)),
            np.sqrt(get_anm(4, 4, cn_table, Y=2.4)),
            np.sqrt(get_anm(5, 5, cn_table, Y=2.4)),
            np.sqrt(get_anm(6, 6, cn_table, Y=2.4)),
            np.sqrt(-get_anm(1, 3, cn_table, Y=2.4)),
            np.sqrt(-get_anm(2, 4, cn_table, Y=2.4)),
            np.sqrt(-get_anm(3, 5, cn_table, Y=2.4)),
            np.sqrt(-get_anm(4, 6, cn_table, Y=2.4)),
            np.sqrt(-get_anm(5, 7, cn_table, Y=2.4))
        ])
        np.savetxt(fname, ann_sqrt)

    plt.plot(ann_sqrt, 'bs')
    plt.xticks(np.arange(0,11), (r'$\sqrt{<a_1^2>}$',
                               r'$\sqrt{<a_2^2>}$',
                               r'$\sqrt{<a_3^2>}$',
                               r'$\sqrt{<a_4^2>}$',
                               r'$\sqrt{<a_5^2>}$',
                               r'$\sqrt{<a_6^2>}$',
                               r'$\sqrt{-<a_1 a_3>}$',
                               r'$\sqrt{-<a_2 a_4>}$',
                               r'$\sqrt{-<a_3 a_5>}$',
                               r'$\sqrt{-<a_4 a_6>}$',
                               r'$\sqrt{-<a_5 a_7>}$'))
    plt.title('Pb+Pb 2.76 TeV, cent %s'%cent)
    plt.xlim(-1, 11)
    plt.ylim(0, 1.2*ann_sqrt.max())

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.show()
