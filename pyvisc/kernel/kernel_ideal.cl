#include<helper.h>

#define ALONG_X 1
#define ALONG_Y 2
#define ALONG_Z 3
// output: d_Src; all the others are input
__kernel void kt_src_alongx(
             __global real4 * d_Src,
		     __global real4 * d_ev,
		     const real tau,
		     const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    if ( I >= NX || J >= NY || K >= NZ ) return;

    int IND = I*NY*NZ + J*NZ + K;
    // store one line of data in local memory
    __local real4 ev[NX+4];
    // load 1D data to local memory
    for ( int i=get_local_id(0); i < NX; i += get_local_size(0) ) {
        ev[i+2] = d_ev[IND];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    // set boundary condition
    if ( (int)get_local_id(0) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NX+3] = ev[NX+1];
       ev[NX+2] = ev[NX+1];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    if ( step == 1 ) d_Src[IND] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    int i = get_local_id(0) + 2;
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
    real4 src4_christoeffel = {Tzz_tilde, 0.0f, 0.0f, Ttz_tilde};

    d_Src[IND] = d_Src[IND] - src4_christoeffel - kt1d(
           ev[i-2], ev[i-1], e_v, ev[i+1], ev[i+2], tau, ALONG_X)/DX;
}

__kernel void kt_src_alongy(
             __global real4 * d_Src,     // out put
		     __global real4 * d_ev,
		     const real tau,
		     const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    if ( I >= NX || J >= NY || K >= NZ ) return;

    int IND = I*NY*NZ + J*NZ + K;
    // store one line of data in local memory
    __local real4 ev[NY+4];

    // load 1D data to local memory
    for ( int i = get_local_id(1); i < NY; i += get_local_size(1) ) {
        ev[i+2] = d_ev[IND];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    // set boundary condition
    if ( get_local_id(1) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NY+3] = ev[NY+1];
       ev[NY+2] = ev[NY+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_local_id(1) + 2;
    d_Src[IND] = d_Src[IND] - kt1d(ev[i-2], ev[i-1],
	  ev[i], ev[i+1], ev[i+2], tau, ALONG_Y)/DY;
}


__kernel void kt_src_alongz(
             __global real4 * d_Src,     // out put
		     __global real4 * d_ev,
		     const real tau,
		     const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    if ( I >= NX || J >= NY || K >= NZ ) return;
    int IND = I*NY*NZ + J*NZ + K;

    // store one line of data in local memory
    __local real4 ev[NZ+4];

    // load 1D data to local memory
    for ( int i = get_local_id(2); i < NZ; i += get_local_size(2) ) {
        ev[i+2] = d_ev[IND];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    // set boundary condition
    if ( get_local_id(2) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NZ+3] = ev[NZ+1];
       ev[NZ+2] = ev[NZ+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_local_id(2) + 2;
    d_Src[IND] = d_Src[IND] - kt1d(ev[i-2], ev[i-1], ev[i], ev[i+1],
				   ev[i+2], tau, ALONG_Z)/(tau*DZ);
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
    real SCALE_COEF = 0.99f;
    if ( M > T00 ) {
	T01 *= SCALE_COEF * T00 / M;
	T02 *= SCALE_COEF * T00 / M;
	T03 *= SCALE_COEF * T00 / M;
        M = SCALE_COEF * T00;
    }

    real ed_find;
    rootFinding_newton(&ed_find, T00, M);
    ed_find = max(acu, ed_find);

    real pr = P(ed_find);

    // vi = T0i/(T00+pr) = (e+p)u0*u0*vi/((e+p)*u0*u0)
    real epv = max(acu, T00 + pr);
    d_evnew[I] = (real4)(ed_find, T01/epv, T02/epv, T03/epv);
}
