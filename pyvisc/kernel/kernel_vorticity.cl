#include<helper.h>

// invariant vorticity vector 
// 2 * omega^{mu nu} = epsilon^{mu nu a b} d_a u_b
// 2 * omega^{tau x} = dyuz - dzuy
// 2 * omega^{tau y} = -(dxuz - dzux)
// 2 * omega^{tau z} = dxuy - dyux
// 2 * omega^{x y} = dtuz - dzut
// 2 * omega^{x z} = -(dtuy - dyut)
// 2 * omega^{y z} = dtux - dxut

// Covariant derivatives == normal derivatives for omega^{mu nu}

// calc beta*u_mu from (ed, vx, vy, tau^2*veta) float4 vector
// (u_t, u_x, u_y, u_eta) where u_eta = - gamma*v_eta
inline real4 ubeta(real4 ev, read_only image2d_t eos_table)
{
    real4 gmn = (real4)(1.0f, -1.0f, -1.0f, -1.0f);
    return gmn*umu4(ev)/T(ev.s0, eos_table);

    // for kinetic vorticity, use the following
    //return gmn*umu4(ev);
}

// wrapper for address index
inline int address(int i, int j, int k)
{
    return i*NY*NZ + j*NZ + k;
}

__kernel void omega(
    __global real4 * d_ev1,
    __global real4 * d_ev2,
	__global real  * d_omega,
    read_only image2d_t eos_table,
	const real tau)
{
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    real4 uold = ubeta(d_ev1[address(I, J, K)], eos_table);
    real4 unew = ubeta(d_ev2[address(I, J, K)], eos_table);

    //   nabla_{t} u_{mu}
    real4 dudt = (unew - uold)/DT;

    real4 dudx = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    if ( I != 0 && I != NX-1 ) {
        dudx = (ubeta(d_ev2[address(I+1, J, K)], eos_table)
              - ubeta(d_ev2[address(I-1, J, K)], eos_table)) / (2.0f*DX);
    } else if ( I == 0 ) { 
        dudx = (ubeta(d_ev2[address(I+1, J, K)], eos_table) - unew) / DX;
    } else if ( I == NX-1 ) {
        dudx = (unew - ubeta(d_ev2[address(I-1, J, K)], eos_table)) / DX;
    }

    real4 dudy = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    if ( J != 0 && J != NY-1 ) {
        dudy = (ubeta(d_ev2[address(I, J+1, K)], eos_table)
              - ubeta(d_ev2[address(I, J-1, K)], eos_table)) / (2.0f*DY);
    } else if ( J == 0 ) { 
        dudy = (ubeta(d_ev2[address(I, J+1, K)], eos_table) - unew) / DY;
    } else if ( J == NY-1 ) {
        dudy = (unew - ubeta(d_ev2[address(I, J-1, K)], eos_table)) / DY;
    }

    // do not use Christoffel symbols, dudz = 1/tau * partial_eta u_{mu}
    // u_{eta} = - gamma*v_eta, has no dimension here

    // real4 dudz = (real4)(unew.s3, 0.0f, 0.0f, -unew.s0)/tau;
    // nabla_{tau} u_{eta} - (1/tau)nabla_{eta}u_{tau} = partial_{tau}u_{eta} - (1/tau)partial_{eta}u_{tau}

    real4 dudz = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    if ( K != 0 && K != NZ-1 ) {
        dudz += (ubeta(d_ev2[address(I, J, K+1)], eos_table)
              - ubeta(d_ev2[address(I, J, K-1)], eos_table)) / (2.0f*DZ*tau);
    } else if ( K == 0 ) { 
        dudz += (ubeta(d_ev2[address(I, J, K+1)], eos_table) - unew) / (DZ*tau);
    } else if ( K == NZ-1 ) {
        dudz += (unew - ubeta(d_ev2[address(I, J, K-1)], eos_table)) / (DZ*tau);
    }

    // hbarc convers 1/(GeV*fm) to dimensionless
    d_omega[6*address(I,J,K)+0] = 0.5f * hbarc*(dudy.s3 - dudz.s2);
    d_omega[6*address(I,J,K)+1] = 0.5f * hbarc*(dudz.s1 - dudx.s3);
    d_omega[6*address(I,J,K)+2] = 0.5f * hbarc*(dudx.s2 - dudy.s1);
    d_omega[6*address(I,J,K)+3] = 0.5f * hbarc*(dudt.s3 - dudz.s0);
    d_omega[6*address(I,J,K)+4] = 0.5f * hbarc*(dudy.s0 - dudt.s2);
    d_omega[6*address(I,J,K)+5] = 0.5f * hbarc*(dudt.s1 - dudx.s0);
}


// output: d_omega_mu = (omega^tau, omega^x, omega^y, omega^eta)
__kernel void omega_mu(
	__global real4 * d_omega_mu,
    __global real4 * d_ev,
	__global real  * d_omega,
    const real efrz,
	const real tau)
{
    int I = get_global_id(0);

    real4 ev = d_ev[I];
    real4 umu = umu4(ev);

    real omega_tx = d_omega[6*I + 0];
    real omega_ty = d_omega[6*I + 1];
    real omega_tz = d_omega[6*I + 2];
    real omega_xy = d_omega[6*I + 3];
    real omega_xz = d_omega[6*I + 4];
    real omega_yz = d_omega[6*I + 5];

    // omega4 = \omega^{\mu} = \Omega^{mu nu} u_{nu} 
    // omega^t = u_t * Omega^{t t} + u_x * Omega^{t x} + u_y * Omega^{t y} + u_z * Omega^{t z}
    // omega^x = u_t * Omega^{x t} + u_x * Omega^{x x} + u_y * Omega^{x y} + u_z * Omega^{x z}
    // omega^y = u_t * Omega^{y t} + u_x * Omega^{y x} + u_y * Omega^{y y} + u_z * Omega^{y z}
    // omega^z = u_t * Omega^{z t} + u_x * Omega^{z x} + u_y * Omega^{z y} + u_z * Omega^{z z}
    real4 omega_mu = (real4)(umu.s0*0.0f - umu.s1*omega_tx - umu.s2 * omega_ty - umu.s3*omega_tz,
                           - umu.s0*omega_tx - umu.s1*0.0f - umu.s2 * omega_xy - umu.s3*omega_xz,
                           - umu.s0*omega_ty + umu.s1*omega_xy - umu.s2 * 0.0f - umu.s3*omega_yz,
                           - umu.s0*omega_tz + umu.s1*omega_xz + umu.s2 * omega_yz - umu.s3*0.0f);

    if ( ev.s0 < 0.001 * efrz ) {
        omega_mu = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
    }

    d_omega_mu[I] = omega_mu;

}
