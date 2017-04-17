#include<helper.h>

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2

// output: d_udx = du/dx
// where u={u0, u1, u2, u3}
__kernel void get_dudx(__global real4 * d_udx,
                       __global real4 * d_ev1,
                       const real tau) {
    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    
    // Use num of threads = BSZ to compute src for NX elements
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // set boundary condition (constant extrapolation)
    if ( get_local_id(0) == 0 ) {
        ev[0] = ev[2];
        ev[1] = ev[2];
        ev[NX+3] = ev[NX+1];
        ev[NX+2] = ev[NX+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    real4 dudx;
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int i = I + 2;
        real4 ev_l = ev[i-1];
        real4 ev_r = ev[i+1];
        real u0_l = gamma(ev_l.s1, ev_l.s2, ev_l.s3);
        real u0_r = gamma(ev_r.s1, ev_r.s2, ev_r.s3);
        dudx = (u0_r*(real4)(1.0f, ev_r.s1, ev_r.s2, ev_r.s3) -
                u0_l*(real4)(1.0f, ev_l.s1, ev_l.s2, ev_l.s3))/(2.0*DX);
        
        int IND = I*NY*NZ + J*NZ + K;
        d_udx[IND] = dudx;
    }
}


// output: d_udy = du/dy
// where u={u0, u1, u2, u3}
__kernel void get_dudy(__global real4 * d_udy,
                       __global real4 * d_ev1,
                           const real tau) {
    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    
    // Use num of threads = BSZ to compute src for NX elements
    for ( int J = get_local_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // set boundary condition (constant extrapolation)
    if ( get_global_id(1) == 0 ) {
        ev[0] = ev[2];
        ev[1] = ev[2];
        ev[NY+3] = ev[NY+1];
        ev[NY+2] = ev[NY+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int j = J + 2;
        real4 ev_l = ev[j-1];
        real4 ev_r = ev[j+1];
        real u0_l = gamma(ev_l.s1, ev_l.s2, ev_l.s3);
        real u0_r = gamma(ev_r.s1, ev_r.s2, ev_r.s3);
        real dudy = (u0_r*(real4)(1.0f, ev_r.s1, ev_r.s2, ev_r.s3) -
                u0_l*(real4)(1.0f, ev_l.s1, ev_l.s2, ev_l.s3))/(2.0*DY);
        
        d_udy[IND] = dudy;
    }
}


// output: d_udz = du/dz = du/(tau*deta), z is for simplicity
// where u={u0, u1, u2, u3}
__kernel void get_dudy(__global real4 * d_udy,
                       __global real4 * d_ev1,
                       const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    
    // Use num of threads = BSZ to compute src for NX elements
    for ( int K = get_local_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // set boundary condition (constant extrapolation)
    if ( get_global_id(2) == 0 ) {
        ev[0] = ev[2];
        ev[1] = ev[2];
        ev[NZ+3] = ev[NZ+1];
        ev[NZ+2] = ev[NZ+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int k = K + 2;
    }
}

