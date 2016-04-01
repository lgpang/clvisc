#/usr/bin/env python
#filename: Create_EdUmuPimn.py
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import numpy as np
import sympy as sym
import pyopencl as cl
import os

def gubser_ed(tau, r, L, lam1):
    ''' energy density for 2nd order viscous gubser solution '''
    return np.power(1 + 0.25*(-L*L + tau*tau - r*r)**2/(L*L*tau*tau), -1.33333333333333 + 1.0/lam1)/np.power(tau, 4)

def gubser_vr(tau, r, L):
    q = 1.0/L
    return 2.0*q*q*tau*r/(1.0+q*q*tau*tau+q*q*r*r)

def gubser_pizz(tau, r, L, lam1):
    return  2*np.power((0.25)*(4*np.power(L, 2)*np.power(tau, 2) + np.power(np.power(L, 2)
            - np.power(tau, 2) + np.power(r, 2), 2))/(np.power(L, 2)*np.power(tau, 2)),
            (-1.33333333333333*lam1 + 1)/lam1)/(lam1*np.power(tau, 6)) ;

def gubser_pixx(tau, x, L, lam1, y=0):
    x[np.abs(x)<1.0E-8] = 1.0E-8
    return -tau**2*((4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)/(4*L**2*tau**2))**(-(1.33333333333333*lam1 - 1)/lam1)*(4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)*(x**2*(L**2 + tau**2 + x**2 + y**2)**2 + y**2*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2))/(lam1*tau**6*(x**2 + y**2)*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2)**2)


##### Use GPU parallel to calculate the ini condition for Ed, u^{mu} and pi^{mu nu} 
def CreateIni(ctx, queue, d_ev, d_pi, tau=1.0, L=10.0, lam1=10.0, NX=201, NY=201, NZ=6,
              DX=0.1, DY=0.1, DZ=0.3, fout='BoWenIni_Lam10_L10_gpu.dat'):
    cwd, cwf = os.path.split(__file__)
    prg_src = open(os.path.join(cwd, '../kernel/kernel_gubser_visc.cl'), 'r').read()
    options = ['-D NX=%s'%NX, '-D NY=%s'%NY, '-D NZ=%s'%NZ, '-D DX=%s'%DX, '-D DY=%s'%DY, '-D DZ=%s'%DZ, '-D tau=%s'%tau, '-D L0=%s'%L, '-D lam1=%s'%lam1]
    print options
    
    prg = cl.Program(ctx, prg_src).build(options=options)
    
    pixx = piyy = pixy = 0.0
    prg.CreateIniCond(queue, (NX,NY,NZ), None, d_ev, d_pi,
                      np.float32(pixx), np.float32(pixy), np.float32(piyy))
    

if __name__ == '__main__':
    #os.environ[ 'PYOPENCL_CTX' ] = '0:1' 
    cwd, cwf = os.path.split(__file__)
    import sys
    sys.path.append(os.path.join(cwd, '..'))
    from config import cfg
    from visc import CLVisc
    cfg.IEOS = 0
    Lam = -10.0
    L = 5.0
    cfg.TAU0 = 1.0
    cfg.NX = 501
    cfg.NY = 501
    cfg.NZ = 1
    cfg.DT = 0.005
    cfg.DX = 0.04
    cfg.DY = 0.04
    cfg.LAM1 = Lam
    cfg.ntskip = 100
    cfg.gubser_visc_test = True
    cfg.save_to_hdf5 = False

    visc = CLVisc(cfg, gpu_id=3)
    ctx = visc.ctx
    queue = visc.queue
    CreateIni(ctx, queue, visc.ideal.d_ev[1], visc.d_pi[1], tau=cfg.TAU0,  L=L, lam1=Lam,
              NX=cfg.NX, NY=cfg.NY, NZ=cfg.NZ, DX=cfg.DX, DY=cfg.DY, DZ=cfg.DZ)

    CreateIni(ctx, queue, visc.ideal.d_ev[2], visc.d_pi[2], tau=cfg.TAU0 + cfg.DT,  L=L, lam1=Lam,
              NX=cfg.NX, NY=cfg.NY, NZ=cfg.NZ, DX=cfg.DX, DY=cfg.DY, DZ=cfg.DZ)


    visc.update_udiff(visc.ideal.d_ev[1], visc.ideal.d_ev[2])

    visc.evolve(max_loops=2200, force_run_to_maxloop=True, save_bulk=False,
                plot_bulk=True, save_hypersf=False, save_pi=True)

    bulk = visc.ideal.bulkinfo
    pimn = visc.pimn_info

    xcent = cfg.NX//2

    import h5py
    h5 = h5py.File('gubser_visc_L1.h5', 'w')
    h5.attrs['lam'] = Lam
    h5.attrs['L'] = L
    h5.attrs['eta_over_s'] = cfg.ETAOS
    h5.attrs['DT'] = cfg.DT
    h5.attrs['DX'] = cfg.DX
    h5.attrs['DY'] = cfg.DY
    h5.attrs['NX'] = cfg.NX
    h5.create_dataset('x', data=bulk.x)

    nstep = 10
    tau_list = np.empty(nstep)
    for i in range(nstep):
        h5.create_dataset('clvisc/ex/%s'%i, data=bulk.ex[i])
        h5.create_dataset('clvisc/vx/%s'%i, data=bulk.vx[i])
        h5.create_dataset('clvisc/pizz/%s'%i, data=pimn.pizz_x[i])
        h5.create_dataset('clvisc/pixx/%s'%i, data=pimn.pixx_x[i])
        tau = 1.0 + i*cfg.ntskip*cfg.DT
        tau_list[i] = tau
        h5.create_dataset('gubser/ex/%s'%i,   data=gubser_ed(tau, bulk.x, L, Lam))
        h5.create_dataset('gubser/pizz/%s'%i, data=tau*tau*gubser_pizz(tau,bulk.x,L, Lam))
        h5.create_dataset('gubser/pixx/%s'%i, data=gubser_pixx(tau,bulk.x,L, Lam))
        h5.create_dataset('gubser/vx/%s'%i, data=gubser_vr(tau, bulk.x, L))
    h5.create_dataset('tau', data=tau_list)
    h5.close()


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    fontsize = 25

    xcent = cfg.NX//2

    for i in range(5):
        ax[0, 0].semilogy(bulk.x, bulk.ex[i], 'k-', label='CLVisc')
        ax[0, 0].semilogy(bulk.x, gubser_ed(1.0 + i*cfg.ntskip*cfg.DT, bulk.x, L, Lam), 'r--', label='Gubser')
        ax[0, 0].set_xlabel(r'$r_{\perp}$', fontsize=25)
        ax[0, 0].set_ylabel(r'$\epsilon$', fontsize=25)
        ax[0, 0].legend(loc='best')
        #ax[0, 0].text(bulk.x[xcent], bulk.ex[xcent], r'$\tau=%s$ fm'%i)

    for i in range(5):
        ax[0, 1].plot(bulk.x, bulk.vx[i],'k-')
        ax[0, 1].plot(bulk.x, gubser_vr(1.0 + i*cfg.ntskip*cfg.DT, bulk.x, L), 'r--')
        ax[0, 1].set_xlabel(r'$r_{\perp}$', fontsize=25)
        ax[0, 1].set_ylabel(r'$v_r$', fontsize=25)

    for i in range(5):
        tau = 1.0 + i*cfg.ntskip*cfg.DT
        ax[1, 0].plot(pimn.x, pimn.pizz_x[i], 'k-')
        ax[1, 0].plot(pimn.x, tau*tau*gubser_pizz(tau, bulk.x, L, Lam), 'r--')
        ax[1, 0].set_xlabel(r'$r_{\perp}$', fontsize=25)
        ax[1, 0].set_ylabel(r'$\tau^2 \pi^{\eta\eta}$', fontsize=25)

    for i in range(5):
        tau = 1.0 + i*cfg.ntskip*cfg.DT
        ax[1, 1].plot(pimn.x, pimn.pixx_x[i], 'k-')
        ax[1, 1].plot(pimn.x, gubser_pixx(tau, bulk.x, L, Lam), 'r--')
        ax[1, 1].set_xlabel(r'$r_{\perp}$', fontsize=25)
        ax[1, 1].set_ylabel(r'$\pi^{xx}$', fontsize=25)

    plt.show()
