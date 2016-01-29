#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Di 26 Jan 2016 12:09:21 CET

import matplotlib.pyplot as plt
import numpy as np

path = '../../results/P30_WB/'
#number of eta (-9.0, 9.0) with deta=0.3
NZ = 61 
# number of x grid, (-15.0, 15.0) with dx=0.1
NX = 301

dx, deta = 0.1, 0.3
x = np.linspace(-(NX-1)/2*dx, (NX-1)/2*dx, NX, endpoint=True)
eta = np.linspace(-(NZ-1)/2*deta, (NZ-1)/2*deta, NZ, endpoint=True)
eta, x = np.meshgrid(eta, x)
tau0, dtau = 0.4, 0.3


def umu(time_step):
    '''get (ut, ux, uy, uz) from (utau, ux', uy', ueta)'''
    vx = np.loadtxt(path+'/vx_xz%s.dat'%time_step)
    # vy is not zero with fluctuating initial conditions
    vy = np.loadtxt(path+'/vy_xz%s.dat'%time_step)
    veta = np.loadtxt(path+'/vz_xz%s.dat'%time_step)
    # constrain veta 
    veta[veta>0.999] = 0.999
    veta[veta<-0.999] = -0.999
    # veta is negative at eta>0 ?
    #plt.imshow(veta, aspect='auto')
    #plt.colorbar()
    Y = np.arctanh(veta) + eta
    coef = np.cosh(Y-eta)/np.cosh(Y)
    
    # vx_ = \tilde{v_x}, ...
    vx_ = vx * coef
    vy_ = vy * coef
    vz_ = np.tanh(Y)
    ut_ = 1.0/np.sqrt(1.0-vx_*vx_-vy_*vy_-vz_*vz_)
    return ut_, -ut_*vx_, -ut_*vy_, -ut_*vz_


def dtau_ux(ux0, ux1):
    '''time derivatives for ux'''
    return (ux1 - ux0)/dtau

def deta_ux(ux1):
    '''ux gradient along eta, 
       axis=0 for x, axis=1 for eta'''
    return np.gradient(ux1)[1]/deta

def dx_uz(uz1):
    '''uz gradient along x, 
       axis=0 for x, axis=1 for eta'''
    return np.gradient(uz1)[0]/dx

def dxuz_dzux(time_step):
    '''dxuz - dzux for the vorticity'''
    ux0, ux1, uz1 = None, None, None
    if time_step == 0:
        ut0, ux0, uy0, uz0 = umu(time_step=0)
        ut1, ux1, uy1, uz1 = umu(time_step=1)
    else:
        ut0, ux0, uy0, uz0 = umu(time_step-1)
        ut1, ux1, uy1, uz1 = umu(time_step)
    
    tau = tau0 + dtau*time_step
        
    dzux = -np.sinh(eta)*dtau_ux(ux0, ux1) \
           + np.cosh(eta)/tau*deta_ux(ux1)
        
    temperature = np.loadtxt(path+'/T_xz%s.dat'%time_step)
        
    return (dx_uz(uz1) - dzux)*temperature, dx_uz(uz1)*temperature, - dzux*temperature



def draw_vorticity_in_reaction_plane(time_step):
    '''make plot for different time step'''
    tau = tau0 + dtau*time_step
    dxuz, minus_dzux, vorticity = dxuz_dzux(time_step)

    extent = (-(NZ-1)/2*deta, (NZ-1)/2*deta, -(NX-1)/2*dx, (NX-1)/2*dx, )
    plt.imshow(vorticity, extent=extent, aspect='auto', origin='lower', vmin=-1., vmax=1.)
    #plt.imshow(vorticity, extent=extent, aspect='auto', origin='lower')
    plt.xlim(-5, 5)
    plt.colorbar()
    plt.xlabel(r'$\eta$', fontsize=25)
    plt.ylabel(r'x [fm]', fontsize=25)
    plt.title(r'$T(\partial_x u_z - \partial_z u_x)$ at $\tau$=%s [fm]'%tau,\
             fontsize=20)
    plt.show()
    

def save_dxuz_dzux(time_step):
    dxuz, minus_dzux, vorticity = dxuz_dzux(time_step)
    np.savetxt('dxuz_%d.dat'%time_step, dxuz)
    np.savetxt('minus_dzux_%d.dat'%time_step, minus_dzux)
    np.savetxt('omega_xz_%d.dat'%time_step, vorticity)


for n in range(16):
    save_dxuz_dzux(time_step=n)

