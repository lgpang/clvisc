#include<helper.h>

#define IDX_PI 11

/* d_SrcT: Src for viscous hydro dT^{tau mu}/dt terms
   d_SrcN: Src for conserved charge densities
   d_ev:   (ed, vx, vy, veta)
   d_nc:   (nb, ne, ns, null)
   d_pi:   (10 for pi^{mu nu} and 1 for PI)
*/
__kernel void kt_src_christoffel(
             __global real4 * d_SrcT,
		     __global real4 * d_ev,
		     __global real * d_pi,
             read_only image2d_t eos_table,
		     const real tau,
             const int step) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
        if ( step == 1 ) {
            d_SrcT[I] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
        real4 e_v = d_ev[I];
        real ed = e_v.s0;
        real vx = e_v.s1;
        real vy = e_v.s2;
        real vz = e_v.s3;
        real u0 = gamma(vx, vy, vz);

        //real bulkpi = d_pi[idn(I, IDX_PI)];
        real bulkpi = 0.0f;
        real pressure = P(ed, eos_table) + bulkpi;

        // Tzz_tilde = T^{eta eta} * tau^2; no 1/tau in vz
        real Tzz_tilde = (ed + pressure)*u0*u0*vz*vz + pressure
                         + d_pi[idn(I, idx(3, 3))];
        real Ttz_tilde = (ed + pressure)*u0*u0*vz
                         + d_pi[idn(I, idx(0, 3))];
        d_SrcT[I] = d_SrcT[I] - (real4)(Tzz_tilde, 0.0f, 0.0f, Ttz_tilde);
    }
}


// kt1d_visc to calc flux residual for T^mn=T_0^{mn} + pi^{mn}
// Qpi_ph = 0.5*(pi^{t mu}_ip1 + pi^{t mu}_i)
// Fpi_ph = 0.5*(pi^{w mu}_ip1 + pi^{w mu}_i) where w=x, y, eta

real4 kt1d_visc(real4 ev[5], real4 pim0[5], real4 pimi[5],
           real vip_half, real vim_half,
           real tau, int along, read_only image2d_t eos_table);

real4 kt1d_visc(real4 ev[5], real4 pim0[5], real4 pimi[5],
           real vip_half, real vim_half,
           real tau, int along, read_only image2d_t eos_table) {
   real pr[5];
   real4 Q[5];
   for ( int i = 0; i < 5; i ++ ) {
       pr[i] = P(ev[i].s0, eos_table);
       Q[i] = tau * (t0m(ev[i], pr[i]) + pim0[i]);
   }

   real4 DA0, DA1;
   int i = 2;
   DA0 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
   i = 3;
   DA1 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
  
   real4  AL = Q[2]   + 0.5f * DA0;
   real4  AR = Q[3] - 0.5f * DA1;

   real pr_half = 0.5f*(pr[2] + pr[3]);
   real4 pim0_half = 0.5f*(pim0[2] + pim0[3]);
   real4 pimi_half = 0.5f*(pimi[2] + pimi[3]);
   // Flux Jp = (T0m + pr*g^{tau mu} - pim0)*v^x - pr*g^{x mu} + pimi
   real4 Jp = (AR + pr_half*tau*gm[0] - tau*pim0_half)*vip_half - pr_half*tau*gm[along+1] + tau*pimi_half;
   real4 Jm = (AL + pr_half*tau*gm[0] - tau*pim0_half)*vip_half - pr_half*tau*gm[along+1] + tau*pimi_half;

   real4 ev_half = 0.5f*(ev[2]+ev[3]);
   // maximum local propagation speed at i+1/2
   real lam = maxPropagationSpeed(ev_half, vip_half, pr_half);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   DA1 = DA0;  // reuse the previous calculate value
   i = 1;
   DA0 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   AL = Q[1]   + 0.5f * DA0;
   AR = Q[2] - 0.5f * DA1;

   // pr_half = tau*pr(i+1/2)
   pr_half = 0.5f*(pr[1] + pr[2]);
   pim0_half = 0.5f*(pim0[1] + pim0[2]);
   pimi_half = 0.5f*(pimi[1] + pimi[2]);

   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   Jp = (AR + pr_half*tau*gm[0] - tau*pim0_half)*vim_half - pr_half*tau*gm[along+1] + tau*pimi_half;
   Jm = (AL + pr_half*tau*gm[0] - tau*pim0_half)*vim_half - pr_half*tau*gm[along+1] + tau*pimi_half;

   // maximum local propagation speed at i-1/2
   ev_half = 0.5f*(ev[2] + ev[1]);
   lam = maxPropagationSpeed(ev_half, vim_half, pr_half);
   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}


// output: d_Src; all the others are input
__kernel void kt_src_alongx(
             __global real4 * d_Src,
		     __global real4 * d_ev,
             __global real  * d_pi,
             read_only image2d_t eos_table,
		     const real tau) {
    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real4 pim0[NX+4];
    __local real4 pimi[NX+4];

    // Use num of threads = BSZ to compute src for NX elements
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
        pim0[I+2] = (real4)(d_pi[idn(IND, 0)],
                            d_pi[idn(IND, 1)],
                            d_pi[idn(IND, 2)],
                            d_pi[idn(IND, 3)]);
        pimi[I+2] = (real4)(d_pi[idn(IND, idx(1, 0))],
                            d_pi[idn(IND, idx(1, 1))],
                            d_pi[idn(IND, idx(1, 2))],
                            d_pi[idn(IND, idx(1, 3))]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_local_id(0) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NX+3] = ev[NX+1];
       ev[NX+2] = ev[NX+1];

       pim0[0] = pim0[2];
       pim0[1] = pim0[2];
       pim0[NX+3] = pim0[NX+1];
       pim0[NX+2] = pim0[NX+1];

       pimi[0] = pimi[2];
       pimi[1] = pimi[2];
       pimi[NX+3] = pimi[NX+1];
       pimi[NX+2] = pimi[NX+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // load the following data from local to private memory
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = I + 2;
        real4 p_ev[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};
        real4 p_pim0[5] ={pim0[i-2], pim0[i-1], pim0[i], pim0[i+1], pim0[i+2]};
        real4 p_pimi[5] ={pimi[i-2], pimi[i-1], pimi[i], pimi[i+1], pimi[i+2]};
        real vip_half = 0.5f*(p_ev[2].s1 + p_ev[3].s1);
        real vim_half = 0.5f*(p_ev[1].s1 + p_ev[2].s1);

        d_Src[IND] = d_Src[IND] - kt1d_visc(p_ev, p_pim0, p_pimi, vip_half,
                             vim_half, tau, ALONG_X, eos_table)/DX;
    }
}

// output: d_Src; all the others are input
__kernel void kt_src_alongy(
             __global real4 * d_Src,
		     __global real4 * d_ev,
             __global real  * d_pi,
             read_only image2d_t eos_table,
		     const real tau) {
    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    __local real4 pim0[NY+4];
    __local real4 pimi[NY+4];

    // Use num of threads = BSZ to compute src for NX elements
    for ( int J = get_local_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
        pim0[J+2] = (real4)(d_pi[idn(IND, 0)],
                            d_pi[idn(IND, 1)],
                            d_pi[idn(IND, 2)],
                            d_pi[idn(IND, 3)]);
        pimi[J+2] = (real4)(d_pi[idn(IND, idx(1, 0))],
                            d_pi[idn(IND, idx(1, 1))],
                            d_pi[idn(IND, idx(1, 2))],
                            d_pi[idn(IND, idx(1, 3))]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_global_id(1) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NY+3] = ev[NY+1];
       ev[NY+2] = ev[NY+1];

       pim0[0] = pim0[2];
       pim0[1] = pim0[2];
       pim0[NY+3] = pim0[NY+1];
       pim0[NY+2] = pim0[NY+1];

       pimi[0] = pimi[2];
       pimi[1] = pimi[2];
       pimi[NY+3] = pimi[NY+1];
       pimi[NY+2] = pimi[NY+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = J + 2;

        real4 p_ev[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};
        real4 p_pim0[5] ={pim0[i-2], pim0[i-1], pim0[i], pim0[i+1], pim0[i+2]};
        real4 p_pimi[5] ={pimi[i-2], pimi[i-1], pimi[i], pimi[i+1], pimi[i+2]};

        real vip_half = 0.5f*(p_ev[2].s2 + p_ev[3].s2);
        real vim_half = 0.5f*(p_ev[1].s2 + p_ev[2].s2);
        d_Src[IND] = d_Src[IND] - kt1d_visc(p_ev, p_pim0, p_pimi, vip_half,
                                 vim_half, tau, ALONG_Y, eos_table)/DY;
    }
}

// output: d_Src; all the others are input
__kernel void kt_src_alongz(
             __global real4 * d_Src,
		     __global real4 * d_ev,
             __global real  * d_pi,
             read_only image2d_t eos_table,
		     const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real4 pim0[NZ+4];
    __local real4 pimi[NZ+4];

    // Use num of threads = BSZ to compute src for NX elements
    for ( int K = get_local_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        pim0[K+2] = (real4)(d_pi[idn(IND, 0)],
                            d_pi[idn(IND, 1)],
                            d_pi[idn(IND, 2)],
                            d_pi[idn(IND, 3)]);
        pimi[K+2] = (real4)(d_pi[idn(IND, idx(3, 0))],
                            d_pi[idn(IND, idx(3, 1))],
                            d_pi[idn(IND, idx(3, 2))],
                            d_pi[idn(IND, idx(3, 3))]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_global_id(2) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NZ+3] = ev[NZ+1];
       ev[NZ+2] = ev[NZ+1];

       pim0[0] = pim0[2];
       pim0[1] = pim0[2];
       pim0[NZ+3] = pim0[NZ+1];
       pim0[NZ+2] = pim0[NZ+1];

       pimi[0] = pimi[2];
       pimi[1] = pimi[2];
       pimi[NZ+3] = pimi[NZ+1];
       pimi[NZ+2] = pimi[NZ+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = K + 2;
        real4 p_ev[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};
        real4 p_pim0[5] ={pim0[i-2], pim0[i-1], pim0[i], pim0[i+1], pim0[i+2]};
        real4 p_pimi[5] ={pimi[i-2], pimi[i-1], pimi[i], pimi[i+1], pimi[i+2]};

        real vip_half = 0.5f*(p_ev[2].s3 + p_ev[3].s3);
        real vim_half = 0.5f*(p_ev[1].s3 + p_ev[2].s3);

        d_Src[IND] = d_Src[IND] - kt1d_visc(p_ev, p_pim0, p_pimi, vip_half,
                         vim_half, tau, ALONG_Z, eos_table)/(tau*DZ);
    }
}


/** update d_evnew with d_ev1 and d_Src*/
__kernel void update_ev(
	__global real4 * d_evnew,
	__global real4 * d_ev1,
    __global real * d_pi1,
    __global real * d_pi2,
	__global real4 * d_Src,
    read_only image2d_t eos_table,
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
    real pressure = P(ed, eos_table);
    real u0 = gamma(vx, vy, vz);
    real4 umu = u0*(real4)(1.0f, vx, vy, vz);
    real4 pim0 = (real4)(d_pi1[idn(I, 0)],
                         d_pi1[idn(I, 1)],
                         d_pi1[idn(I, 2)],
                         d_pi1[idn(I, 3)]);

    // when step=2, tau=(n+1)*DT, while T0m need tau=n*DT
    real old_time = tau - (step-1)*DT;
    real4 T0m = ((ed + pressure)*u0*umu - pressure*gm[0] + pim0)
                * old_time;

    /** step==1: Q' = Q0 + Src*DT
        step==2: Q  = Q0 + (Src(Q0)+Src(Q'))*DT/2
    */
    pim0 = (real4)(d_pi2[idn(I, 0)], d_pi2[idn(I, 1)],
                         d_pi2[idn(I, 2)], d_pi2[idn(I, 3)]);

    T0m = T0m + d_Src[I]*DT/step - pim0*tau;

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
        //M = 0.0f;
    }

    real ed_find;
    rootFinding_newton(&ed_find, T00, M, eos_table);
    //rootFinding(&ed_find, T00, M, eos_table);
    ed_find = max(0.0f, ed_find);

    real pr = P(ed_find, eos_table);

    // vi = T0i/(T00+pr) = (e+p)u0*u0*vi/((e+p)*u0*u0)
    real epv = max(acu, T00 + pr);
    d_evnew[I] = (real4)(ed_find, T01/epv, T02/epv, T03/epv);
    }
}
