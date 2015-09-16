#ifndef __HELPERH__
#define __HELPERH__
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

//#pragma OPENCL EXTENSION cl_amd_printf: enable

#include"real_type.h"
#include"EosPCEv0.cl"

#define theta 1.1f
#define hbarc 0.19733f
#define acu 1.0E-8f

#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))

// kt1d to calc H(i+1/2)-H(i-1/2), along=1,2,3 for x, y, z
real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1, real4 ev_ip2, real tau, int along);


// g^{tau mu}, g^{x mu}, g^{y mu}, g^{eta mu} without tau*tau
constant real4 gm[4] = 
{(real4)(1.0f, 0.0f, 0.0f, 0.0f),
(real4)(0.0f, -1.0f, 0.0f, 0.0f),
(real4)(0.0f, 0.0f, -1.0f, 0.0f),
(real4)(0.0f, 0.0f, 0.0f, -1.0f)};
//////////////////////////////////////////////////////////
/*!Cacl gamma from vx, vy, vz where vz=veta in Milne space */
inline real gamma(real vx, real vy, real vz){
    return 1.0f/sqrt(1.0f-vx*vx-vy*vy-vz*vz);
}

/** 1D linear interpolation */
inline real lin_int( real x1, real x2, real y1, real y2, real x )
{
    real r = ( x - x1 )/ (x2-x1 );
    return y1*(1.0f-r) + y2*r ;
}

inline real poly3_int( real x1, real x2, real x3, real y1, real y2, real y3, real x )
{
  return y1*(x-x2)*(x-x3)/((x1-x2)*(x1-x3)) + y2*(x-x1)*(x-x3)/((x2-x1)*(x2-x3)) + y3*(x-x1)*(x-x2)/((x3-x1)*(x3-x2)) ;
}


/** Flux limit for scalar and real4*/
inline real minmod(real x, real y) {
    real res = min(fabs(x), fabs(y));
    return res*(sign(x)+sign(y))*0.5f;
}

inline real4 minmod4(real4 x, real4 y) {
    real4 res = min(fabs(x), fabs(y));
    return res*(sign(x)+sign(y))*0.5f;
}

/** Calc maximum local propagation speed along k direction*/
inline real maxPropagationSpeed(real4 edv, real vk, real pr){
    real ut = gamma(edv.s1, edv.s2, edv.s3);
    real uk = ut*vk;
    real ut2 = ut*ut;
    real uk2 = uk*uk;
    real cs2 = pr/edv.s0;
    real lam = (fabs(ut*uk*(1.0f-cs2))+sqrt((ut2-uk2-(ut2-uk2-1.0f)*cs2)*cs2))
       /(ut2 - (ut2-1.0f)*cs2);
    return max(lam, 1.33f);
}


/** solve energy density from T00 and K=sqrt(T01**2 + T02**2 + T03**2)
 * */
inline void rootFinding( real* EdFind, real * T00, real *K2 ){
    real E0, E1;
    E1 = *T00;   /*predict */
    int i = 0;
    while ( true ) {
        E0 = E1;
        E1 = *T00 - (*K2)/( *T00 + P(E1)) ; 
        i ++ ;
        if( i>20 || fabs(E1-E0)/max(fabs(E1), (real)acu)<acu ) break;
    }

    * EdFind = E1;
}


/** construct \tilde{T}^{mu *} */
inline real4 Tmu(real4 edv, real pr, int mu) {
    real u0 = gamma(edv.s1, edv.s2, edv.s3);
    real ed = edv.s0;
    real4 u4 = (real4)(1.0f, edv.s1, edv.s2, edv.s3)*u0;
    return (ed+pr)*u4[mu]*u4 - gm[mu]*pr;
}

real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1,
           real4 ev_ip2, real tau, int along) {
   real pr_im1 = P(ev_im1.s0);
   real pr_i = P(ev_i.s0);
   real pr_ip1 = P(ev_ip1.s0);
   real pr_ip2 = P(ev_ip2.s0);

   real4 T0m_im1 = tau*Tmu(ev_im1, pr_im1, 0);
   real4 T0m_i = tau*Tmu(ev_i, pr_i, 0);
   real4 T0m_ip1 = tau*Tmu(ev_ip1, pr_ip1, 0);
   real4 T0m_ip2 = tau*Tmu(ev_ip2, pr_ip2, 0);

   real4 DA0, DA1;
   DA0 = minmod4(0.5f*(T0m_ip1-T0m_im1),
           minmod4(theta*(T0m_ip1-T0m_i), theta*(T0m_i-T0m_im1)));

   DA1 = minmod4(0.5f*(T0m_ip2-T0m_i),
         minmod4(theta*(T0m_ip2-T0m_ip1), theta*(T0m_ip1-T0m_i)));
  
   real vim1[4] = {1.0f, ev_im1.s1, ev_im1.s2, ev_im1.s3};
   real vi[4] = {1.0f, ev_i.s1, ev_i.s2, ev_i.s3};
   real vip1[4] = {1.0f, ev_ip1.s1, ev_ip1.s2, ev_ip1.s3};

   real lam = maxPropagationSpeed(ev_i, vi[along], pr_i);

   real4  AL = T0m_i   + 0.5f * DA0;
   real4  AR = T0m_ip1 - 0.5f * DA1;

   real pr_half = lin_int(0.0f, 1.0f, pr_i, pr_ip1, 0.5f) * tau;
   // Flux Jp = (T0m + pr*g^{tau mu})*v^x + pr*g^{x mu}
   real vi_half = lin_int(0.0f, 1.0f, vi[along], vip1[along], 0.5f);
   real4 Jp = (AR + pr_half*gm[0])*vi_half + pr_half*gm[along];
   real4 Jm = (AL + pr_half*gm[0])*vi_half + pr_half*gm[along];

   // first part of kt1d; the final results = src[i]-src[i-1]
   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   real pr_im2 = P(ev_im2.s0);
   real4 T0m_im2 = tau*Tmu(ev_im2, pr_im2, 0);
   DA1 = DA0;  // reuse the previous calculate value
   DA0 = minmod4(0.5f*(T0m_i-T0m_im2),
           minmod4(theta*(T0m_i-T0m_im1), theta*(T0m_im1-T0m_im2)));
   lam = maxPropagationSpeed(ev_im1, vim1[along], pr_im1);
   AL = T0m_im1 + 0.5f * DA0;
   AR = T0m_i - 0.5f * DA1;

   // pr_half = tau*pr(i+1/2)
   pr_half = lin_int(0.0f, 1.0f, pr_im1, pr_i, 0.5f)*tau;
   // Flux Jp = (T0m + pr*g^{tau mu})*v^x + pr*g^{x mu}
   vi_half = lin_int(0.0f, 1.0f, vim1[along], vi[along], 0.5f);
   Jp = (AR + pr_half*gm[0])*vi_half + pr_half*gm[along];
   Jm = (AL + pr_half*gm[0])*vi_half + pr_half*gm[along];

   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}

#endif
