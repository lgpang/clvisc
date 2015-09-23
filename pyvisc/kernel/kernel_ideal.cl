#include<helper.h>

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2


// output: d_Src; all the others are input
__kernel void kt_src(
             __global real4 * d_Src,
		     __global real4 * d_ev,
		     const real tau,
		     const int step,
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
    if ( get_group_id(direction) != 0 && i==2 ) {
        if ( direction == ALONG_X ) {
            ev[0] = d_ev[(I-2)*NY*NZ+J*NZ+K];
            ev[1] = d_ev[(I-1)*NY*NZ+J*NZ+K];
        } else if ( direction == ALONG_Y ) {
            ev[0] = d_ev[I*NY*NZ+(J-2)*NZ+K];
            ev[1] = d_ev[I*NY*NZ+(J-1)*NZ+K];
        } else if ( direction == ALONG_Z ) {
            ev[0] = d_ev[I*NY*NZ+J*NZ+K-2];
            ev[1] = d_ev[I*NY*NZ+J*NZ+K-1];
        }
    }
    
    // set boundary condition
    if ( get_group_id(direction) != get_num_groups(direction)
        && i == BSZ+1 ) {
        if ( direction == ALONG_X ) {
            ev[BSZ+3] = d_ev[(I+2)*NY*NZ+J*NZ+K];
            ev[BSZ+2] = d_ev[(I+1)*NY*NZ+J*NZ+K];
        } else if ( direction == ALONG_Y ) {
            ev[BSZ+3] = d_ev[I*NY*NZ+(J+2)*NZ+K];
            ev[BSZ+2] = d_ev[I*NY*NZ+(J+1)*NZ+K];
        } else if ( direction == ALONG_Z ) {
            ev[BSZ+3] = d_ev[I*NY*NZ+J*NZ+K+2];
            ev[BSZ+2] = d_ev[I*NY*NZ+J*NZ+K+1];
        }
    }
    
    if ( get_global_id(direction) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
    }
    
    int NSize[3] = {NX, NY, NZ};
    if ( get_global_id(direction) == NSize[direction] ) {
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


/** update d_evnew with d_ev1 and d_Src*/
__kernel void update_ev(
	__global real4 * d_evnew,
	__global real4 * d_ev1,
	__global real4 * d_Src,
	const real tau,
	const int  step)
{
    int I = get_global_id(0);
    if ( I < NX*NY*NZ ) {
    real4 e_v = d_ev1[I];
    real ed = e_v.s0;
    real vx = e_v.s1;
    real vy = e_v.s2;
    real vz = e_v.s3;
    real pressure = P(ed);
    real u0 = gamma(vx, vy, vz);
    real4 umu = u0*(real4)(1.0f, vx, vy, vz);

    // when step=2, tau=(n+1)*DT, while T0m need tau=n*DT
    real old_time = tau - (step-1)*DT;
    real4 T0m = ((ed + pressure)*u0*umu - pressure*gm[0])
                * old_time;

    /** step==1: Q' = Q0 + Src*DT
        step==2: Q  = Q0 + (Src(Q0)+Src(Q'))*DT/2
    */
    T0m = T0m + d_Src[I]*DT/step;

    real T00 = max(acu, T0m.s0)/tau;
    real T01 = (fabs(T0m.s1) < acu) ? 0.0f : T0m.s1/tau;
    real T02 = (fabs(T0m.s2) < acu) ? 0.0f : T0m.s2/tau;
    real T03 = (fabs(T0m.s3) < acu) ? 0.0f : T0m.s3/tau;

    real M = sqrt(T01*T01 + T02*T02 + T03*T03);
    real SCALE_COEF = 0.999f;
    if ( M > T00 ) {
	    T01 *= SCALE_COEF * T00 / M;
	    T02 *= SCALE_COEF * T00 / M;
	    T03 *= SCALE_COEF * T00 / M;
        M = SCALE_COEF * T00;
    }

    real ed_find;
    rootFinding_newton(&ed_find, T00, M);
    //rootFinding(&ed_find, T00, M);
    ed_find = max(acu, ed_find);

    real pr = P(ed_find);

    // vi = T0i/(T00+pr) = (e+p)u0*u0*vi/((e+p)*u0*u0)
    real epv = max(acu, T00 + pr);
    d_evnew[I] = (real4)(ed_find, T01/epv, T02/epv, T03/epv);
    }
}
