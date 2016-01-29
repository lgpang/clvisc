#include<helper.h>

// invariant vorticity vector 
// omega^{mu} = epsilon^{mu nu a b} u_{nu} d_a u_b
// for mu = tau,
// omega^{tau} = ux dy ueta - ux deta uy - uy dx ueta + uy deta ux  + ueta dx uy - ueta dy ux
// omega^{x} = 
__kernel void omegamu(
    __global real4 * d_ev1,
    __global real4 * d_ev2,
    __global real4 * d_udiff,
    __global real4 * d_udx,
    __global real4 * d_udy,
    __global real4 * d_udz,
	__global real4 * d_omegamu,
    read_only image2d_t eos_table,
	const real tau,
	const int  step)
{
    int I = get_global_id(0);
    
    real4 e_v1 = d_ev1[I];
    real4 e_v2 = d_ev2[I];
    real4 u_old = umu4(e_v1);
    real4 u_new = umu4(e_v2);
    real4 udt = (u_new - u_old)/DT;
    // correct with previous udiff=u_visc-u_ideal*
    //if ( step == 1 ) udt += d_udiff[I]/DT;
    if ( step == 1 ) { 
        udt = d_udiff[I]/DT;
        u_new = u_old + udt;
    }

    real4 udx = d_udx[I];
    real4 udy = d_udy[I];
    real4 udz = d_udz[I];

    // omega0 = -omega^{tau}
    real omega0 = u_new.s1*(udy.s3 - udz.s2) 
                + u_new.s2*(udz.s1 - udx.s3) 
                + u_new.s3*(udx.s2 - udy.s1);

    real omega1 = u_new.s0*(udz.s2 - udy.s3) 
                + u_new.s2*(udt.s3 - udz.s0) 
                + u_new.s3*(udy.s0 - udt.s2);

    real omega2 = u_new.s0*(- udz.s1 + udx.s3) 
                + u_new.s1*(- udt.s3 + udz.s0) 
                + u_new.s3*(- udx.s0 + udt.s1);

    real omega3 = u_new.s0*(udx.s2 - udy.s1) 
                + u_new.s1*(udy.s0 - udt.s2) 
                + u_new.s2*(udt.s1 - udx.s0);

    d_omegamu[I] = (real4)(-omega0, omega1, omega2, omega3);

}
