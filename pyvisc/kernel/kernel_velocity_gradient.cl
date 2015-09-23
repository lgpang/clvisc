#include<helper.h>

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2

// output: d_nabla_u[16]
__kernel void get_nabla_u(
             __global real4 * d_ev0,
		     __global real4 * d_ev1,
             __global real * d_nabla_t_u,
		     const real tau,
             const int direction) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    int IND = I*NY*NZ + J*NZ + K;
    int i = get_local_id(direction) + 2;
    __local real4 ev[BSZ+4];

    bool in_range = (I < NX && J < NY && K < NZ);

    if ( in_range ) { ev[i] = d_ev[IND]; }

    // barrier can not be in the if( in_range ) because that 
    // threads with I>NX or J>NY or K>NZ never reach this point
    // and deadlock is created in CLK_LOCAL_MEM_FENCE
    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition
    if ( i == 2 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[BSZ+3] = ev[BSZ+1];
       ev[BSZ+2] = ev[BSZ+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ( in_range ) {
        real4 christoffel_src = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
        if ( direction == ALONG_X ) {
            if ( step == 1 ) {
               d_Src[IND] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
            }
            real4 e_v = ev[i];
            real ed = e_v.s0;
            real vx = e_v.s1;
            real vy = e_v.s2;
            real vz = e_v.s3;
            real pressure = P(ed);
            real u0 = gamma(vx, vy, vz);

            // Tzz_tilde = T^{eta eta} * tau^2; no 1/tau in vz
            real Tzz_tilde = (ed + pressure)*u0*u0*vz*vz + pressure;
            real Ttz_tilde = (ed + pressure)*u0*u0*vz;
            christoffel_src = (real4)(Tzz_tilde, 0.0f, 0.0f, Ttz_tilde);
        }

        real DW[3] = {DX, DY, DZ*tau};
        d_Src[IND] = d_Src[IND] - christoffel_src - kt1d(ev[i-2], ev[i-1],
                     ev[i], ev[i+1], ev[i+2], tau, direction)/DW[direction];
    }
}


