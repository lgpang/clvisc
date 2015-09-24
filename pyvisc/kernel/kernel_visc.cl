#include<helper.h>

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2

// use pimn and ev at  i-2, i-1, i, i+1, i+2 to calc src term from flux
// pr_mh = pr_{i-1/2} and pr_ph = pr_{i+1/2}
// these mh, ph terms are calcualted 1 time and used 10 times by pimn
real kt1d_pimn(
       real pi_im2, real pi_im1, real pi_i, real pi_ip1, real pi_ip2,
       real u0_im2, real u0_im1, real u0_i, real u0_ip1, real u0_ip2,
       real pr_mh, real pr_ph, real v_mh, real v_ph, real lam_mh, real lam_ph,
       real tau, int along)
{
   real T0m_im1 = u0_im1 * pi_im1;
   real T0m_i = u0_i * pi_i;
   real T0m_ip1 = u0_ip1 * pi_ip1;
   real T0m_ip2 = u0_ip2 * pi_ip2;

   real DA0, DA1;
   DA0 = minmod(0.5f*(T0m_ip1-T0m_im1),
           minmod(theta*(T0m_ip1-T0m_i), theta*(T0m_i-T0m_im1)));

   DA1 = minmod(0.5f*(T0m_ip2-T0m_i),
         minmod(theta*(T0m_ip2-T0m_ip1), theta*(T0m_ip1-T0m_i)));

   real  AL = T0m_i   + 0.5f * DA0;
   real  AR = T0m_ip1 - 0.5f * DA1;

   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   real Jp = AR * v_ph;
   real Jm = AL * v_ph;

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam_ph*(AR-AL);

   real T0m_im2 = u0_im2 * pi_im2;
   DA1 = DA0;  // reuse the previous calculate value
   DA0 = minmod(0.5f*(T0m_i-T0m_im2),
           minmod(theta*(T0m_i-T0m_im1), theta*(T0m_im1-T0m_im2)));

   AL = T0m_im1 + 0.5f * DA0;
   AR = T0m_i - 0.5f * DA1;

   Jp = AR*v_mh;
   Jm = AL*v_mh;

   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam_mh*(AR-AL);

   return src;
}


// christoffel terms from d;mu pi^{\alpha \beta}
__kernel void visc_src_christoffel(
             __global real * d_Src,
             __global real * d_pi1,
		     __global real4 * d_ev,
		     const real tau,
             const int step) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
        if ( step == 1 ) {
           for ( int mn = 0; mn < 10; mn ++ ) {
               d_Src[10*I+mn] = 0.0f;
           }
        }
        
        real4 e_v = d_ev[I];
        real uz = e_v.s3 * gamma(e_v.s1, e_v.s2, e_v.s3);

        d_Src[10*I+0] += 2.0f * uz * d_pi1[idx(0, 3)]/tau;
        d_Src[10*I+1] += uz * d_pi1[idx(1, 3)]/tau;
        d_Src[10*I+2] += uz * d_pi1[idx(2, 3)]/tau;
        d_Src[10*I+3] += uz * (d_pi1[idx(0, 0)] + d_pi1[idx(3,3)])/tau;
        // d_Src[10*I+4] += 0.0f;
        // d_Src[10*I+5] += 0.0f;
        d_Src[10*I+6] += uz * d_pi1[idx(0, 1)]/tau;
        // d_Src[10*I+7] += 0.0f;
        d_Src[10*I+8] += uz * d_pi1[idx(0, 2)]/tau;
        d_Src[10*I+9] += 2.0f * uz * d_pi1[idx(0, 3)]/tau;
    }
}

// output: d_Src kt src for pimn evolution;
// output: d_udx the velocity gradient along x
// all the others are input
__kernel void visc_src_alongx(
             __global real * d_Src,
             __global real4 * d_udx,
             __global real * d_pi1,
		     __global real4 * d_ev,
		     const real tau) {
    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real pimn[10*(NX+4)];

    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
        int startI = 10*I + 2;
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[startI + mn] = d_pi1[10*IND + mn];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_local_id(0) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NX+3] = ev[NX+1];
       ev[NX+2] = ev[NX+1];
       for ( int mn = 0; mn < 10; mn ++ ) {
             pimn[mn] = pimn[20+mn];
             pimn[10+mn] = pimn[20+mn];
             pimn[(NX+3)*10+mn] = pimn[(NX+1)*10+mn];
             pimn[(NX+2)*10+mn] = pimn[(NX+1)*10+mn];
       }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = I + 2;
        real4 ev_im2 = ev[i-2];
        real4 ev_im1 = ev[i-1];
        real4 ev_i   = ev[i];
        real4 ev_ip1 = ev[i+1];
        real4 ev_ip2 = ev[i+2];
        

        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        real pr_mh = 0.5f*(P(ev_im1.s0) + P(ev_i.s0));
        real pr_ph = 0.5f*(P(ev_ip1.s0) + P(ev_i.s0));
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s1 + ev_i.s1);
        real v_ph = 0.5f*(ev_ip1.s1 + ev_i.s1);
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i), v_mh, pr_mh);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i), v_ph, pr_ph);
        
        d_udx[IND] = (u0_ip1*(real4)(1.0f, ev_ip1.s1, ev_ip1.s2, ev_ip1.s3)
            -u0_im1*(real4)(1.0f, ev_im1.s1, ev_im1.s2, ev_im1.s3))/(2.0f*DX);
        
        for ( int mn = 0; mn < 10; mn ++ ) {
            d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_pimn(
               pimn[(i-2)*10+mn], pimn[(i-1)*10+mn], pimn[i*10+mn],
               pimn[(i+1)*10+mn], pimn[(i+2)*10+mn],
               u0_im2, u0_im1, u0_i, u0_ip1, u0_ip2,
               pr_mh, pr_ph, v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_X)/DX;
        }
    }
}


// output: d_Src kt src for pimn evolution;
// output: d_udy the velocity gradient along x
// all the others are input
__kernel void visc_src_alongy(
                              __global real * d_Src,
                              __global real4 * d_udy,
                              __global real * d_pi1,
                              __global real4 * d_ev,
                              const real tau) {
    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    __local real pimn[10*(NY+4)];
    
    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
        int startJ = 10*J + 2;
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[startJ + mn] = d_pi1[10*IND + mn];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // set boundary condition (constant extrapolation)
    if ( get_local_id(1) == 0 ) {
        ev[0] = ev[2];
        ev[1] = ev[2];
        ev[NY+3] = ev[NY+1];
        ev[NY+2] = ev[NY+1];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[mn] = pimn[20+mn];
            pimn[10+mn] = pimn[20+mn];
            pimn[(NY+3)*10+mn] = pimn[(NY+1)*10+mn];
            pimn[(NY+2)*10+mn] = pimn[(NY+1)*10+mn];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = J + 2;
        real4 ev_im2 = ev[i-2];
        real4 ev_im1 = ev[i-1];
        real4 ev_i   = ev[i];
        real4 ev_ip1 = ev[i+1];
        real4 ev_ip2 = ev[i+2];
        
        
        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        real pr_mh = 0.5f*(P(ev_im1.s0) + P(ev_i.s0));
        real pr_ph = 0.5f*(P(ev_ip1.s0) + P(ev_i.s0));
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s2 + ev_i.s2);
        real v_ph = 0.5f*(ev_ip1.s2 + ev_i.s2);
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i), v_mh, pr_mh);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i), v_ph, pr_ph);
        
        d_udy[IND] = (u0_ip1*(real4)(1.0f, ev_ip1.s1, ev_ip1.s2, ev_ip1.s3)
                      -u0_im1*(real4)(1.0f, ev_im1.s1, ev_im1.s2, ev_im1.s3))/(2.0f*DY);
        
        for ( int mn = 0; mn < 10; mn ++ ) {
            d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_pimn(
                        pimn[(i-2)*10+mn], pimn[(i-1)*10+mn], pimn[i*10+mn],
                        pimn[(i+1)*10+mn], pimn[(i+2)*10+mn],
                        u0_im2, u0_im1, u0_i, u0_ip1, u0_ip2,
                        pr_mh, pr_ph, v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Y)/DY;
        }
    }
}



// output: d_Src kt src for pimn evolution;
// output: d_udy the velocity gradient along x
// all the others are input
__kernel void visc_src_alongz(
                              __global real * d_Src,
                              __global real4 * d_udz,
                              __global real * d_pi1,
                              __global real4 * d_ev,
                              const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real pimn[10*(NZ+4)];
    
    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int K = get_global_id(0); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        int startK = 10*K + 2;
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[startK + mn] = d_pi1[10*IND + mn];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // set boundary condition (constant extrapolation)
    if ( get_local_id(1) == 0 ) {
        ev[0] = ev[2];
        ev[1] = ev[2];
        ev[NZ+3] = ev[NZ+1];
        ev[NZ+2] = ev[NZ+1];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[mn] = pimn[20+mn];
            pimn[10+mn] = pimn[20+mn];
            pimn[(NZ+3)*10+mn] = pimn[(NZ+1)*10+mn];
            pimn[(NZ+2)*10+mn] = pimn[(NZ+1)*10+mn];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = K + 2;
        real4 ev_im2 = ev[i-2];
        real4 ev_im1 = ev[i-1];
        real4 ev_i   = ev[i];
        real4 ev_ip1 = ev[i+1];
        real4 ev_ip2 = ev[i+2];
        
        
        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        real pr_mh = 0.5f*(P(ev_im1.s0) + P(ev_i.s0));
        real pr_ph = 0.5f*(P(ev_ip1.s0) + P(ev_i.s0));
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s3 + ev_i.s3);
        real v_ph = 0.5f*(ev_ip1.s3 + ev_i.s3);
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i), v_mh, pr_mh);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i), v_ph, pr_ph);
        
        real4 christoffel_term = (real4)(ev_i.s3, 0.0f, 0.0f, 1.0f)*u0_i/tau;
        d_udz[IND] = (u0_ip1*(real4)(1.0f, ev_ip1.s1, ev_ip1.s2, ev_ip1.s3)
                      -u0_im1*(real4)(1.0f, ev_im1.s1, ev_im1.s2, ev_im1.s3))/(2.0f*tau*DZ)
                      + christoffel_term;
        
        for ( int mn = 0; mn < 10; mn ++ ) {
            d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_pimn(
                  pimn[(i-2)*10+mn], pimn[(i-1)*10+mn], pimn[i*10+mn],
                  pimn[(i+1)*10+mn], pimn[(i+2)*10+mn],
                  u0_im2, u0_im1, u0_i, u0_ip1, u0_ip2,
                  pr_mh, pr_ph, v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Y)/(tau*DZ);
        }
    }
}


