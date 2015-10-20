#ifndef __HELPERH__
#define __HELPERH__

#include"real_type.h"
//#include"EosPCEv0.cl"
#include"eos_table.h"

#define THETA 1.1f
#define hbarc 0.19733f

// represent 16 components of pi^{mu nu} in 10 elements
#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))

// kt1d to calc H(i+1/2)-H(i-1/2), along=0,1,2 for x, y, z
real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1, real4 ev_ip2,
           real tau, int along, read_only image2d_t eos_table);

// g^{tau mu}, g^{x mu}, g^{y mu}, g^{eta mu} without tau*tau
constant real4 gm[4] = 
{(real4)(1.0f, 0.0f, 0.0f, 0.0f),
(real4)(0.0f, -1.0f, 0.0f, 0.0f),
(real4)(0.0f, 0.0f, -1.0f, 0.0f),
(real4)(0.0f, 0.0f, 0.0f, -1.0f)};
//////////////////////////////////////////////////////////
/*!Cacl gamma from vx, vy, vz where vz=veta in Milne space */
inline real gamma(real vx, real vy, real vz){
    return 1.0f/sqrt(max(1.0f-vx*vx-vy*vy-vz*vz, acu));
}

/*!Cacl gamma from (real4)(ed, vx, vy, vz) where vz=veta in Milne space */
inline real gamma_real4(real4 ev){
    return 1.0f/sqrt(max(1.0f-ev.s1*ev.s1-ev.s2*ev.s2-ev.s3*ev.s3, acu));
}

/** get (real4) umu4 from (real4) ev */
inline real4 umu4(real4 ev){
    return gamma_real4(ev)*(real4)(1.0, ev.s123);
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

// Calc du/dt, du/dx or du/dy or du/dz where du/dz = du/(tau deta)
inline real4 dudw(real4 ul, real4 ur, real dw) {
    return (ur - ul)/dw;
}


/** Calc maximum propagation speed along k direction
 * The maximum lam can not be bigger than 1.0f, in relativity
 * if fluid velocity is v=1, maximum cs2=1/3, the signal speed
 * in computing frame is cs2' = (cs2+v)/(1+cs2*v) = 1.0 */
inline real maxPropagationSpeed(real4 edv, real vk, real pr){
    real ut = gamma(edv.s1, edv.s2, edv.s3);
    real uk = ut*vk;
    real ut2 = ut*ut;
    real uk2 = uk*uk;
    real cs2 = pr/max(edv.s0, acu);
    real lam = (fabs(ut*uk*(1.0f-cs2))+sqrt((ut2-uk2-(ut2-uk2-1.0f)*cs2)*cs2))
       /(ut2 - (ut2-1.0f)*cs2);
    return max(lam, 1.0f);
}


/** solve energy density from T00 and K=sqrt(T01**2 + T02**2 + T03**2)
 * The rootfinding can not pass tests for many events
 * 74% fail if absolute error < 1.0E-6
 * 20% fail if absolute error < 1.0E-3
 * 0.001% fail if absolute error < 0.01
 * What happens to these testing events?
 * */
inline void rootFinding(real* EdFind, real T00, real M, read_only image2d_t eos_table){
    real E0, E1;
    E1 = T00;   /*predict */
    real K2 = M*M;
    int i = 0;
    while ( true ) {
        E0 = E1;
        E1 = T00 - K2/(T00 + P(E1, eos_table)) ; 
        i ++ ;
        if( i>100 || fabs(E1-E0)/max(fabs(E1), (real)acu)<acu ) break;
    }

    * EdFind = E1;
}

// bisection and newton method for the root finding
inline void rootFinding_newton(real* ed_find, real T00, real M,
                               read_only image2d_t eos_table){
             
    real vl = 0.0f;
    real vh = 1.0f;
    real dpe = 1.0f/3.0f;
    real v = 0.5f*(vl+vh);
    real ed = T00 - M*v;
    real pr = P(ed, eos_table);
    real f = (T00 + pr)*v - M;
    real df = (T00+pr) - M*v*dpe;
    real dvold = vh-vl;
    real dv = dvold;
    int i = 0;
    while ( true ) {
        if ((f + df * (vh - v)) * (f + df * (vl - v)) > 0.0f ||
            fabs(2.f * f) > fabs(dvold * df)) {  // bisection
          dvold = dv;
          dv = 0.5f * (vh - vl);
          v = vl + dv;
        } else {  // Newton
          dvold = dv;
          dv = f / df;
          v -= dv;
        }
        i ++;
        if ( fabs(dv) < 0.00001f || i > 100 ) break;

        ed = T00 - M*v;
        pr = P(ed, eos_table);
        f = (T00 + pr)*v - M;
        df = (T00+pr) - M*v*dpe;
        if ( f > 0.0f ) {
             vh = v;
        } else { 
             vl = v;
        }
    }
    * ed_find = T00 - M*v;
}



/** construct T^{tau mu} 4 vector*/
inline real4 t0m(real4 edv, real pr) {
    real u0 = gamma(edv.s1, edv.s2, edv.s3);
    real ed = edv.s0;
    real4 u4 = (real4)(1.0f, edv.s1, edv.s2, edv.s3)*u0;
    return (ed+pr)*u0*u4 - gm[0]*pr;
}

real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1,
           real4 ev_ip2, real tau, int along,
           read_only image2d_t eos_table) {
   real pr_im1 = P(ev_im1.s0, eos_table);
   real pr_i = P(ev_i.s0, eos_table);
   real pr_ip1 = P(ev_ip1.s0, eos_table);
   real pr_ip2 = P(ev_ip2.s0, eos_table);

   real4 T0m_im1 = tau*t0m(ev_im1, pr_im1);
   real4 T0m_i = tau*t0m(ev_i, pr_i);
   real4 T0m_ip1 = tau*t0m(ev_ip1, pr_ip1);
   real4 T0m_ip2 = tau*t0m(ev_ip2, pr_ip2);

   real4 DA0, DA1;
   DA0 = minmod4(0.5f*(T0m_ip1-T0m_im1),
           minmod4(THETA*(T0m_ip1-T0m_i), THETA*(T0m_i-T0m_im1)));

   DA1 = minmod4(0.5f*(T0m_ip2-T0m_i),
         minmod4(THETA*(T0m_ip2-T0m_ip1), THETA*(T0m_ip1-T0m_i)));
  
   real vim1[3] = {ev_im1.s1, ev_im1.s2, ev_im1.s3};
   real vi[3] = {ev_i.s1, ev_i.s2, ev_i.s3};
   real vip1[3] = {ev_ip1.s1, ev_ip1.s2, ev_ip1.s3};

   real4  AL = T0m_i   + 0.5f * DA0;
   real4  AR = T0m_ip1 - 0.5f * DA1;

   real pr_half = 0.5f*(pr_ip1 + pr_i);
   real vi_half = 0.5f*(vi[along] + vip1[along]);
   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   real4 Jp = (AR + pr_half*tau*gm[0])*vi_half - pr_half*tau*gm[along+1];
   real4 Jm = (AL + pr_half*tau*gm[0])*vi_half - pr_half*tau*gm[along+1];

   real4 ev_half = 0.5f*(ev_i+ev_ip1);
   // maximum local propagation speed at i+1/2
   real lam = maxPropagationSpeed(ev_half, vi_half, pr_half);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   real pr_im2 = P(ev_im2.s0, eos_table);
   real4 T0m_im2 = tau*t0m(ev_im2, pr_im2);
   DA1 = DA0;  // reuse the previous calculate value
   DA0 = minmod4(0.5f*(T0m_i-T0m_im2),
           minmod4(THETA*(T0m_i-T0m_im1), THETA*(T0m_im1-T0m_im2)));

   AL = T0m_im1 + 0.5f * DA0;
   AR = T0m_i - 0.5f * DA1;

   // pr_half = tau*pr(i+1/2)
   pr_half = 0.5f*(pr_im1 + pr_i);
   vi_half = 0.5f*(vim1[along] + vi[along]);
   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   Jp = (AR + pr_half*tau*gm[0])*vi_half - pr_half*tau*gm[along+1];
   Jm = (AL + pr_half*tau*gm[0])*vi_half - pr_half*tau*gm[along+1];

   // maximum local propagation speed at i-1/2
   ev_half = 0.5f*(ev_i+ev_im1);
   lam = maxPropagationSpeed(ev_half, vi_half, pr_half);
   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}

#endif
