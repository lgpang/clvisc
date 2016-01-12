// this kernel will be used in kernel_hypersf.cl to get the vorticity omega_{xz}
// on the freeze out hyper surface
//

#include<real_type.h>

/** u_x in (t,x,y,z) coordinates **/
real ux(real4 ev, real etas) {
    real Y = atanh(ev.s3) + etas;
    real vz = tanh(Y);
    real coef = cosh(Y-etas)/cosh(Y);
    real vx = ev.s1 * coef;
    real vy = ev.s2 * coef;
    real gamma = 1.0f/sqrt(max(1.0f - vx*vx - vy*vy - vz*vz, 1.0E-9));

    return -gamma * vx;
}

/** u_z in (t,x,y,z) coordinates **/
real uz(real4 ev, real etas) {
    real Y = atanh(ev.s3) + etas;
    real vz = tanh(Y);
    real coef = cosh(Y-etas)/cosh(Y);
    real vx = ev.s1 * coef;
    real vy = ev.s2 * coef;
    real gamma = 1.0f/sqrt(max(1.0f - vx*vx - vy*vy - vz*vz, 1.0E-9));

    return -gamma * vz;
}


/** calculate omega_{xz} = dxu_z - dzu_x
 * where dzu_x = -sinh eta du_x/dtau + sinh eta / tau * du_x/deta
 * and dxuz = dxuz
 * notice ux, uz here are in (t,x,y,z) coordinates **/
real omega_xz(real4 ev_old, real4 ev_new,
              real4 ev_im1, real4 ev_ip1,
              real4 ev_km1, real4 ev_kp1,
              real tau, real etas) {
    real dxuz = (uz(ev_ip1, etas)-uz(ev_im1, etas))/(2.0f*DX);
    real dtau_ux = (ux(ev_new, etas) - ux(ev_old, etas))/DT;
    real deta_ux = (ux(ev_kp1, etas) - ux(ev_km1, etas))/DZ;
    real dzux = -sinh(etas)*dtau_ux + cosh(etas)/tau*deta_ux;
    real dxuz = (uz(ev_ip1, etas)-uz(ev_im1, etas))/(2.0f*DX);
    return dxuz - dzux;
}

