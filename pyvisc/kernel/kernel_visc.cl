#include<helper.h>

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2

real kt1d_pimn(
       real pi_im2, real pi_im1, real pi_i, real pi_ip1, real pi_ip2,
       real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1, real4 ev_ip2,
       real tau, int along)
{
   real pr_im1 = P(ev_im1.s0);
   real pr_i = P(ev_i.s0);
   real pr_ip1 = P(ev_ip1.s0);
   real pr_ip2 = P(ev_ip2.s0);

   real T0m_im1 = u0_im1 * pi_im1;
   real T0m_i = u0_i * pi_i;
   real T0m_ip1 = u0_ip1 * pi_ip1;
   real T0m_ip2 = u0_ip2 * pi_ip2;

   real DA0, DA1;
   DA0 = minmod(0.5f*(T0m_ip1-T0m_im1),
           minmod4(theta*(T0m_ip1-T0m_i), theta*(T0m_i-T0m_im1)));

   DA1 = minmod(0.5f*(T0m_ip2-T0m_i),
         minmod4(theta*(T0m_ip2-T0m_ip1), theta*(T0m_ip1-T0m_i)));
  
   real vim1[3] = {ev_im1.s1, ev_im1.s2, ev_im1.s3};
   real vi[3] = {ev_i.s1, ev_i.s2, ev_i.s3};
   real vip1[3] = {ev_ip1.s1, ev_ip1.s2, ev_ip1.s3};

   real  AL = T0m_i   + 0.5f * DA0;
   real  AR = T0m_ip1 - 0.5f * DA1;

   real vi_half = 0.5f*(vi[along] + vip1[along]);
   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   real Jp = AR * vi_half;
   real Jm = AL * vi_half;

   real pr_half = 0.5f*(pr_i + pr_ip1);
   real4 ev_half = 0.5f*(ev_i+ev_ip1);
   // maximum local propagation speed at i+1/2
   real lam = maxPropagationSpeed(ev_half, vi_half, pr_half);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   real pr_im2 = P(ev_im2.s0);
   real T0m_im2 = u0_im2 * pi_im2;
   DA1 = DA0;  // reuse the previous calculate value
   DA0 = minmod4(0.5f*(T0m_i-T0m_im2),
           minmod4(theta*(T0m_i-T0m_im1), theta*(T0m_im1-T0m_im2)));

   AL = T0m_im1 + 0.5f * DA0;
   AR = T0m_i - 0.5f * DA1;

   // pr_half = tau*pr(i+1/2)
   pr_half = 0.5f*(pr_im1 + pr_i);
   vi_half = 0.5f*(vim1[along] + vi[along]);
   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   Jp = AR*vi_half;
   Jm = AL*vi_half;

   // maximum local propagation speed at i-1/2
   ev_half = 0.5f*(ev_i+ev_im1);
   lam = maxPropagationSpeed(ev_half, vi_half, pr_half);
   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}


// christoffel terms from d;mu pi^{\alpha \beta}
__kernel void kt_visc_christoffel(
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

// output: d_Src kt src for pimn evolution; all the others are input
__kernel void visc_src_alongx(
             __global real * d_Src,
             __global real * d_pi1,
		     __global real4 * d_ev,
		     const real tau) {
    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real pimn[10*(NX+4)];

    // Use num of threads = BSZ to compute src for NX elements
    // every 10 pimn terms are stored continuesly
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
        for ( int mn = 0; mn < 10; mn ++ ) {
            int i = I + 2;
            d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_pimn(
               pimn[(i-2)*10+mn], pimn[(i-1)*10+mn], pimn[i*10+mn],
               pimn[(i+1)*10+mn], pimn[(i+2)*10+mn], 
               ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2], tau, ALONG_X)/DX;
        }
    }
}



