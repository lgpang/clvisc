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

##### Calc the limit of pixx, pixy, piyy at (x->0, y->0 )
def GetLimit(tau_input=1.0, L_input=10.0, lam1_input=10.0):
    '''pixx, piyy, pixy is not numerically calculable at x->0 and y->0 due to 
       sin(x)/x like structure. '''
    tau, x, y, etas, L, lam1 = sym.symbols('tau x y etas L lam1' )
    
    eps, ut, ux, uy, pitt, pitx, pity, pixx, pixy, piyy, pizz = \
            sym.symbols( 'eps ut ux uy pitt pitx pity pixx pixy piyy pizz' )
    
    pixx= -tau_input**2*((4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)/(4*L**2*tau**2))**(-(1.33333333333333*lam1 - 1)/lam1)*(4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)*(x**2*(L**2 + tau**2 + x**2 + y**2)**2 + y**2*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2))/(lam1*tau**6*(x**2 + y**2)*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2)**2)
    pixy= tau_input**2*x*y*((4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)/(4*L**2*tau**2))**((-1.33333333333333*lam1 + 1)/lam1)*(4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2 - (L**2 + tau**2 + x**2 + y**2)**2)/(lam1*tau**6*(x**2 + y**2)*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2)**2)

    piyy= -tau_input**2*((4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)/(4*L**2*tau**2))**(-(1.33333333333333*lam1 - 1)/lam1)*(4*L**2*tau**2 + (L**2 - tau**2 + x**2 + y**2)**2)*(x**2*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2) + y**2*(L**2 + tau**2 + x**2 + y**2)**2)/(lam1*tau**6*(x**2 + y**2)*(4*L**2*(x**2 + y**2) + (L**2 + tau**2 - x**2 - y**2)**2)**2)

    pixx_limit = sym.limit( pixx.subs( tau, tau_input ).subs(y, 0).subs(L, L_input ).subs( lam1, lam1_input ), x, 0 )
    pixy_limit = sym.limit( pixy.subs( tau, tau_input ).subs(y, 0).subs(L, L_input ).subs( lam1, lam1_input ), x, 0 )
    piyy_limit = sym.limit( piyy.subs( tau, tau_input ).subs(y, 0).subs(L, L_input ).subs( lam1, lam1_input ), x, 0 )

    return pixx_limit, pixy_limit, piyy_limit



##### Use GPU parallel to calculate the ini condition for Ed, u^{mu} and pi^{mu nu} 
def CreateIni(ctx, queue, d_ev, d_pi, tau=1.0, L=10.0, lam1=10.0, NX=201, NY=201, NZ=6,
              DX=0.1, DY=0.1, DZ=0.3, fout='BoWenIni_Lam10_L10_gpu.dat'):
    pixx, pixy, piyy = GetLimit( tau, L, lam1 )
    cwd, cwf = os.path.split(__file__)
    prg_src = open(os.path.join(cwd, '../kernel/kernel_gubser_visc.cl'), 'r').read()
    options = ['-D NX=%s'%NX, '-D NY=%s'%NY, '-D NZ=%s'%NZ, '-D DX=%s'%DX, '-D DY=%s'%DY, '-D DZ=%s'%DZ, '-D tau=%s'%tau, '-D L=%s'%L, '-D lam1=%s'%lam1]
    print options
    
    prg = cl.Program(ctx, prg_src).build(options=options)
    
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
    cfg.NX = 405
    cfg.NY = 405
    cfg.NZ = 1
    cfg.DT = 0.005
    cfg.DX = 0.05
    cfg.DY = 0.05
    cfg.LAM1 = Lam
    cfg.ntskip = 10
    cfg.gubser_visc_test = True
    visc = CLVisc(cfg)
    ctx = visc.ctx
    queue = visc.queue
    CreateIni(ctx, queue, visc.ideal.d_ev[1], visc.d_pi[1], tau=cfg.TAU0,  L=L, lam1=Lam,
              NX=cfg.NX, NY=cfg.NY, NZ=cfg.NZ, DX=cfg.DX, DY=cfg.DY, DZ=cfg.DZ)

    CreateIni(ctx, queue, visc.ideal.d_ev[2], visc.d_pi[2], tau=cfg.TAU0 + cfg.DT,  L=L, lam1=Lam,
              NX=cfg.NX, NY=cfg.NY, NZ=cfg.NZ, DX=cfg.DX, DY=cfg.DY, DZ=cfg.DZ)


    visc.update_udiff(visc.ideal.d_ev[1], visc.ideal.d_ev[2])

    visc.evolve(max_loops=200, force_run_to_maxloop=True, save_bulk=False,
                plot_bulk=True, save_hypersf=False, save_pi=True)

    bulk = visc.ideal.bulkinfo

    import matplotlib.pyplot as plt
    q = 1/L
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(10):
        ax[0].plot(bulk.x, bulk.ex[i])
        ax[0].plot(bulk.x, gubser_ed(1.0 + i*cfg.ntskip*cfg.DT, bulk.x, L, Lam), '--')

    for i in range(10):
        ax[1].plot(bulk.x, bulk.vx[i])
        ax[1].plot(bulk.x, gubser_vr(1.0 + i*cfg.ntskip*cfg.DT, bulk.x, L), '--')

    plt.show()
