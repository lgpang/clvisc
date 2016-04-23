#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy import integrate
from common_plotting import smash_style
from four_momentum import NPHI, pt_integral, phi_integral, PHI, PT
import cmath
from numba import jit

import matplotlib.ticker as mtick


def get_event_plane(dNdPtdPhi2D):
    '''get the event plane from spectra from one rapidity slice'''
    spec_along_phi = np.zeros(NPHI)
    for j in range(NPHI):
        spec_along_pt = dNdPtdPhi2D[:,j]
        spec_along_phi[j] = pt_integral(spec_along_pt)

    Vn = np.zeros(7)
    event_plane = np.zeros(7)
    norm = phi_integral(spec_along_phi)
    for n in xrange(1, 7):
        Vn[n], event_plane[n] = cmath.polar(phi_integral(
                    spec_along_phi*np.exp(1j*n*PHI))/norm)
        event_plane[n] /= float(n)

    return event_plane


def pt_int(spec):
    '''integrate the spec along pt and give dN/dYdPhi '''
    spec_along_phi = np.zeros(NPHI)
    for j in range(NPHI):
        spec_along_pt = spec[:,j]
        spec_along_phi[j] = pt_integral(spec_along_pt)
    return spec_along_phi


def corr(pix, piy, piz):
    '''Pi(phi1) \cdot Pi(phi2) as a function of Delta Phi'''
    newphi = np.concatenate((PHI - 2 * np.pi, PHI, PHI + 2 * np.pi))
    newpix = np.concatenate((pix, pix, pix))
    newpiy = np.concatenate((piy, piy, piy))
    newpiz = np.concatenate((piz, piz, piz))
    fpix = interp1d(newphi, newpix)
    fpiy = interp1d(newphi, newpiy)
    fpiz = interp1d(newphi, newpiz)

    delta_phi = np.linspace(0.0, np.pi, 40, endpoint=True)

    polar_corr_xy = np.zeros_like(delta_phi)
    polar_corr_z = np.zeros_like(delta_phi)

    phi_list = np.linspace(0.0, np.pi*2.0, 79, endpoint=True)

    pix_list = fpix(phi_list)
    piy_list = fpiy(phi_list)
    piz_list = fpiz(phi_list)

    for i, dphi in enumerate(delta_phi):
        pix_dot = (pix_list * (np.roll(pix_list, i) + np.roll(pix_list, -i))).mean()
        piy_dot = (piy_list * (np.roll(piy_list, i) + np.roll(piy_list, -i))).mean()
        piz_dot = (piz_list * (np.roll(piz_list, i) + np.roll(piz_list, -i))).mean()
        polar_corr_xy[i] = pix_dot + piy_dot 
        polar_corr_z[i] = piz_dot

    #polar_avg2 = (pix_list**2 + piy_list**2 + piz_list**2).mean()

    return polar_corr_xy, polar_corr_z


def rapidity_integral(pi3d, Y0, Y1, rapidity):
    '''Params:
    :param pi3d: Pi^{mu} as a function of rapidity, pt, phi
    '''
    Yi = np.linspace(Y0, Y1, 5, endpoint=True)

    def integY(pimu):
        '''integrate over one rapidity slice'''
        return np.interp(Yi, rapidity, pimu).mean()

    rapidity_axis = 0
    pi2d = np.apply_along_axis(func1d=integY, axis=rapidity_axis, arr=pi3d)

    return pi2d
 
def get_polar_in_rapidity(h5, event_id, Y0, Y1):
    '''get polarization vector for particles in rapidity range [Y0, Y1]
    Params:
        :param h5: hdf5 file, which stores mom/Y, mom/PT, mom/PHI, Pi^y and rho
        :param event_id: type int, event id
        :param Y0: type float, lower bound for the rapidity window
        :param Y1: type float, upper bound for the rapidity window
    Returns:
        Pi^{x}, Pi^{y}, Pi^{z} as a function of (pt, phi) for particles in rapidity [Y0, Y1]
        '''
    rapidity = h5['mom/Y'][...]
    pt = h5['mom/PT'][...]
    phi = h5['mom/PHI'][...]

    NY, NPT, NPHI = len(rapidity), len(pt), len(phi)
    # 5.0 is a typo when create the dataset
    name = 'event%s/rapidity5.0/pol_vs_pt_phi'%(event_id)
    pol = h5[name][...]
    name = 'event%s/rapidity5.0/rho_vs_pt_phi'%(event_id)
    rho = h5[name][...]

    rho0 = rho.reshape(NY, NPT, NPHI)
    pix = pol[:, 1].reshape(NY, NPT, NPHI)
    piy = pol[:, 2].reshape(NY, NPT, NPHI)
    piz = pol[:, 3].reshape(NY, NPT, NPHI)

   
    # exchange Y0, Y1 if Y0 > Y1
    if Y0 > Y1: Y0, Y1 = Y1, Y0

    pix = rapidity_integral(pix, Y0, Y1, rapidity)
    piy = rapidity_integral(piy, Y0, Y1, rapidity)
    piz = rapidity_integral(piz, Y0, Y1, rapidity)
    rho = rapidity_integral(rho0, Y0, Y1, rapidity)

    return pix, piy, piz, rho


#@profile
def azimuthal_correlation(withz=True, etaos='0p08', system='auau62p4', n=80, Y0=-0.5, Y1=0.5):
    count = 0
    with h5py.File('vor_int_visc%s_%s_cent20_30.hdf5'%(etaos, system), 'r') as h5:
        rapidity = h5['mom/Y'][...]
        pt = h5['mom/PT'][...]
        phi = h5['mom/PHI'][...]

        NY, NPT, NPHI = len(rapidity), len(pt), len(phi)

        # bug in the creating h5 scritp
        Y = rapidity[-1]

        pol_mean = np.zeros(NPHI)

        # correlation Pi(p1) \cdot Pi(p2) as a function of delta_phi
        count = 0

        delta_phi = np.linspace(0.1, np.pi, 40, endpoint=True)
        pol_corr_xy = np.zeros_like(delta_phi)
        pol_corr_z = np.zeros_like(delta_phi)

        start_id = 0
        end_id = 200
        if system == 'auau200':
            start_id = 300
            end_id = 650

        for event_id in range(start_id, end_id):
            try:
                pix, piy, piz, rho = get_polar_in_rapidity(h5, event_id, Y0, Y1)

                if not withz:
                    piz = np.zeros_like(piy)

                rho_int = pt_int(rho)
                pix_int = pt_int(pix) / rho_int
                piy_int = pt_int(piy) / rho_int
                piz_int = pt_int(piz) / rho_int

                if not np.isnan(piy[0, 0]):
                    corri, corrj = corr(pix_int, piy_int, piz_int)
                    pol_corr_xy += corri
                    pol_corr_z += corrj
                    count += 1

                print event_id, ' finished!'
            except:
                print event_id, ' does not exist!'

        pol_corr_xy /= count
        pol_corr_z /= count

        print('num of good events is:', count)
        maxi = pol_corr_xy.max()
        mini = pol_corr_xy.min()
        maxf = max(abs(maxi), abs(mini))*1.2

        if withz:
            maxi = pol_corr_z.max()
            mini = pol_corr_z.min()
            maxf = max(abs(maxi), abs(mini))*1.2

        return maxf, delta_phi, pol_corr_xy, pol_corr_z



def plot_range(Y0, Y1, withz, system, etaos, color='grey'):
    max0, delta_phi, pol_corr1, pol_corr_z1 = azimuthal_correlation(withz=withz, etaos=etaos, system=system, n=650, Y0= -Y1, Y1= -Y0)
    max0, delta_phi, pol_corr2, pol_corr_z2 = azimuthal_correlation(withz=withz, etaos=etaos, system=system, n=650, Y0= Y0, Y1=Y1)
    if not withz:
        plt.plot(delta_phi, 0.5*(pol_corr1 + pol_corr2), color=color, label=r'$|Y|=[%s, %s]$'%(Y0, Y1))
        plt.fill_between(delta_phi, pol_corr1, pol_corr2, facecolor=color, alpha=0.5)
    else:
        plt.plot(delta_phi, 0.5*(pol_corr_z1 + pol_corr_z2), color=color, label=r'$|Y|=[%s, %s]$'%(Y0, Y1))
        plt.fill_between(delta_phi, pol_corr_z1, pol_corr_z2, facecolor=color, alpha=0.5)
    return max0

def plot_pol_corr(withz=True, system='auau200', etaos='0p08'):
    max0 = plot_range(0, 1, withz, system, etaos, color='red')
    max1 = plot_range(1, 2, withz, system, etaos, color='blue')
    max2 = plot_range(2, 3, withz, system, etaos, color='green')

    max3 = max0
    if system == 'pbpb2p76':
        max3 = plot_range(4, 5, withz, system, etaos, color='grey')

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    plt.xlabel(r'$|\phi_1 - \phi_2|$')
    maxf = max(max0, max1, max2, max3)
    plt.ylim(-maxf, maxf)
    smash_style.set()
    plt.legend(loc='best')
    plt.subplots_adjust(left=0.2)

    system_title = '$Au+Au\ 200\ GeV$,'
    if system == 'pbpb2p76':
        system_title = '$Pb+Pb\ 2.76\ TeV$,'
    if system == 'auau62p4':
        system_title = '$Au+Au\ 62.4\ GeV$,'

    fname = ''
    if withz:
        plt.ylabel(r'$<\Pi_{\eta}(\phi_1) \cdot \Pi_{\eta}(\phi_2)>$')

        fname = 'figs/Piz_corr_vs_deltaphi_{system}_{etaos}.pdf'.format(system=system, etaos=etaos)
        plt.title('{system_title} 20-30%, $\eta/s$={etaos}'.format(
            system_title=system_title, etaos=etaos.replace('p', '.') ))
    else:
        plt.ylabel(r'$<\Pi_{\perp}(\phi_1) \cdot \Pi_{\perp}(\phi_2)>$')

        fname = 'figs/Pixy_corr_vs_deltaphi_{system}_{etaos}.pdf'.format(system=system, etaos=etaos)
        plt.title('{system_title} 20-30%, $\eta/s$={etaos}'.format(
            system_title=system_title, etaos=etaos.replace('p', '.') ))
    plt.savefig(fname)
    plt.close()


#plot_pol_corr(withz=False, system='auau200', etaos='0p0')
#plot_pol_corr(withz=False, system='auau200', etaos='0p08')
#plot_pol_corr(withz=True, system='auau200', etaos='0p0')
#plot_pol_corr(withz=True, system='auau200', etaos='0p08')

plot_pol_corr(withz=False, system='pbpb2p76', etaos='0p0')
plot_pol_corr(withz=False, system='pbpb2p76', etaos='0p02')
plot_pol_corr(withz=False, system='pbpb2p76', etaos='0p04')
plot_pol_corr(withz=False, system='pbpb2p76', etaos='0p08')
plot_pol_corr(withz=False, system='pbpb2p76', etaos='0p12')
plot_pol_corr(withz=False, system='pbpb2p76', etaos='0p16')

plot_pol_corr(withz=False, system='auau62p4', etaos='0p08')
plot_pol_corr(withz=True, system='auau62p4', etaos='0p08')

##
##plot_pol_corr(withz=True, system='auau200', etaos='0p12')
##plot_pol_corr(withz=True, system='auau62p4', etaos='0p08')
plot_pol_corr(withz=True, system='pbpb2p76', etaos='0p0')
plot_pol_corr(withz=True, system='pbpb2p76', etaos='0p02')
plot_pol_corr(withz=True, system='pbpb2p76', etaos='0p04')
plot_pol_corr(withz=True, system='pbpb2p76', etaos='0p08')
plot_pol_corr(withz=True, system='pbpb2p76', etaos='0p12')
plot_pol_corr(withz=True, system='pbpb2p76', etaos='0p16')
