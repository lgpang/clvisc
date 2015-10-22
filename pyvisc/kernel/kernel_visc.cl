#include<helper.h>

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2

// IDN() return the idx of pi^{mu nu}_{i,j,k}in global mem
// I = i*NY*NZ + j*NZ + k
#define idn(I, mn) (I)*10+mn

constant real gmn[4][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                           {0.0f,-1.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f,-1.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f,-1.0f}};

// use pimn and ev at  i-2, i-1, i, i+1, i+2 to calc src term from flux
// pr_mh = pr_{i-1/2} and pr_ph = pr_{i+1/2}
// these mh, ph terms are calcualted 1 time and used 10 times by pimn
real kt1d_real(
       real Q_im2, real Q_im1, real Q_i, real Q_ip1, real Q_ip2,
       real pr_mh, real pr_ph, real v_mh, real v_ph, real lam_mh, real lam_ph,
       real tau, int along)
{
   real DA0, DA1;
   DA0 = minmod(0.5f*(Q_ip1-Q_im1),
           minmod(THETA*(Q_ip1-Q_i), THETA*(Q_i-Q_im1)));

   DA1 = minmod(0.5f*(Q_ip2-Q_i),
         minmod(THETA*(Q_ip2-Q_ip1), THETA*(Q_ip1-Q_i)));

   real  AL = Q_i   + 0.5f * DA0;
   real  AR = Q_ip1 - 0.5f * DA1;

   // Flux Jp = (Q + pr*g^{tau mu})*v^x - pr*g^{x mu}
   real Jp = AR * v_ph;
   real Jm = AL * v_ph;

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam_ph*(AR-AL);

   DA1 = DA0;  // reuse the previous calculate value
   DA0 = minmod(0.5f*(Q_i-Q_im2),
           minmod(THETA*(Q_i-Q_im1), THETA*(Q_im1-Q_im2)));

   AL = Q_im1 + 0.5f * DA0;
   AR = Q_i - 0.5f * DA1;

   Jp = AR*v_mh;
   Jm = AL*v_mh;

   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam_mh*(AR-AL);

   return src;
}


// christoffel terms from d;mu pi^{\alpha \beta}
__kernel void visc_initialize(
             __global real * d_pi1,
		     __global real4 * d_ev,
		     const real tau,
             read_only image2d_t eos_table) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
       real4 ev = d_ev[I];
       real etav = ETAOS * S(ev.s0, eos_table) * hbarc;
       real tmp = 2.0f/3.0f * etav / tau;
       //d_pi1[10*I+idx(1, 1)] = tmp;
       //d_pi1[10*I+idx(2, 2)] = tmp;
       //d_pi1[10*I+idx(3, 3)] = -2.0f*tmp;
       d_pi1[10*I+idx(1, 1)] = 0.0f;
       d_pi1[10*I+idx(2, 2)] = 0.0f;
       d_pi1[10*I+idx(3, 3)] = 0.0f;
    }
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
             read_only image2d_t eos_table,
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
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(I+2, mn)] = d_pi1[idn(IND, mn)];
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
             pimn[idn(0, mn)] = pimn[idn(2, mn)];
             pimn[idn(1, mn)] = pimn[idn(2, mn)];
             pimn[idn(NX+3, mn)] = pimn[idn(NX+1, mn)];
             pimn[idn(NX+2, mn)] = pimn[idn(NX+1, mn)];
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
        real pr_mh = 0.5f*(P(ev_im1.s0, eos_table) + P(ev_i.s0, eos_table));
        real pr_ph = 0.5f*(P(ev_ip1.s0, eos_table) + P(ev_i.s0, eos_table));
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s1 + ev_i.s1);
        real v_ph = 0.5f*(ev_ip1.s1 + ev_i.s1);
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i), v_mh, pr_mh);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i), v_ph, pr_ph);
        
        //d_udx[IND] = (u0_ip1*(real4)(1.0f, ev_ip1.s123)
        //    -u0_im1*(real4)(1.0f, ev_im1.s123))/(2.0f*DX);
        
        d_udx[IND] = minmod4(0.5f*(umu4(ev_ip1)-umu4(ev_im1)),
                             minmod4(THETA*(umu4(ev_i) - umu4(ev_im1)),
                                     THETA*(umu4(ev_ip1) - umu4(ev_i))
                                     ))/DX;

        for ( int mn = 0; mn < 10; mn ++ ) {
            d_Src[idn(IND, mn)] = d_Src[idn(IND, mn)] - kt1d_real(
               u0_im2*pimn[idn(i-2, mn)], u0_im1*pimn[idn(i-1, mn)],
               u0_i*pimn[idn(i, mn)], u0_ip1*pimn[idn(i+1, mn)],
               u0_ip2*pimn[idn(i+2, mn)],
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
                              read_only image2d_t eos_table,
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
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(J+2, mn)] = d_pi1[idn(IND, mn)];
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
            pimn[idn(0, mn)] = pimn[idn(2, mn)];
            pimn[idn(1, mn)] = pimn[idn(2, mn)];
            pimn[idn(NY+3, mn)] = pimn[idn(NY+1, mn)];
            pimn[idn(NY+2, mn)] = pimn[idn(NY+1, mn)];
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
        real pr_mh = 0.5f*(P(ev_im1.s0, eos_table) + P(ev_i.s0, eos_table));
        real pr_ph = 0.5f*(P(ev_ip1.s0, eos_table) + P(ev_i.s0, eos_table));
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s2 + ev_i.s2);
        real v_ph = 0.5f*(ev_ip1.s2 + ev_i.s2);
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i), v_mh, pr_mh);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i), v_ph, pr_ph);
        
        //d_udy[IND] = (u0_ip1*(real4)(1.0f, ev_ip1.s123)
        //              -u0_im1*(real4)(1.0f, ev_im1.s123))/(2.0f*DY);

        d_udy[IND] = minmod4(0.5f*(umu4(ev_ip1)-umu4(ev_im1)),
                             minmod4(THETA*(umu4(ev_i) - umu4(ev_im1)),
                                     THETA*(umu4(ev_ip1) - umu4(ev_i))
                                     ))/DY;


        for ( int mn = 0; mn < 10; mn ++ ) {
            d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_real(
               u0_im2*pimn[(i-2)*10+mn], u0_im1*pimn[(i-1)*10+mn],
               u0_i*pimn[i*10+mn], u0_ip1*pimn[(i+1)*10+mn],
               u0_ip2*pimn[(i+2)*10+mn],
               pr_mh, pr_ph, v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Y)/DY;
        }
    }
}



// output: d_Src kt src for pimn evolution;
// output: d_udy the velocity gradient along x
// all the others are input
__kernel void visc_src_alongz(__global real * d_Src,
                              __global real4 * d_udz,
                              __global real * d_pi1,
                              __global real4 * d_ev,
                              read_only image2d_t eos_table,
                              const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real pimn[10*(NZ+4)];
    
    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(K+2, mn)] = d_pi1[idn(IND, mn)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // set boundary condition (constant extrapolation)
    if ( get_local_id(2) == 0 ) {
        ev[0] = ev[2];
        ev[1] = ev[2];
        ev[NZ+3] = ev[NZ+1];
        ev[NZ+2] = ev[NZ+1];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[mn] = pimn[20+mn];
            pimn[10+mn] = pimn[20+mn];
            pimn[idn(NZ+3, mn)] = pimn[idn(NZ+1, mn)];
            pimn[idn(NZ+2, mn)] = pimn[idn(NZ+1, mn)];
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
        real pr_mh = 0.5f*(P(ev_im1.s0, eos_table) + P(ev_i.s0, eos_table));
        real pr_ph = 0.5f*(P(ev_ip1.s0, eos_table) + P(ev_i.s0, eos_table));
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s3 + ev_i.s3);
        real v_ph = 0.5f*(ev_ip1.s3 + ev_i.s3);
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i), v_mh, pr_mh);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i), v_ph, pr_ph);
        
        real4 christoffel_term = (real4)(ev_i.s3, 0.0f, 0.0f, 1.0f)*u0_i/tau;
        
        //d_udz[IND] = (u0_ip1*(real4)(1.0f, ev_ip1.s123)
        //              -u0_im1*(real4)(1.0f, ev_im1.s123))/(2.0f*tau*DZ)
        //              + christoffel_term;

        d_udz[IND] = minmod4(0.5f*(umu4(ev_ip1)-umu4(ev_im1)),
                             minmod4(THETA*(umu4(ev_i) - umu4(ev_im1)),
                                     THETA*(umu4(ev_ip1) - umu4(ev_i))
                            ))/(DZ*tau) + christoffel_term;


        for ( int mn = 0; mn < 10; mn ++ ) {
            d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_real(
               u0_im2*pimn[(i-2)*10+mn], u0_im1*pimn[(i-1)*10+mn],
               u0_i*pimn[i*10+mn], u0_ip1*pimn[(i+1)*10+mn],
               u0_ip2*pimn[(i+2)*10+mn],
               pr_mh, pr_ph, v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Z)/(tau*DZ);
        }
    }
}

/** update d_pinew with d_pi1 and d_Src*/
__kernel void update_pimn(
	__global real4 * d_checkpi,
	__global real * d_pinew,
	__global real * d_pi1,
    __global real4 * d_ev1,
    __global real4 * d_ev2,
    __global real4 * d_udx,
    __global real4 * d_udy,
    __global real4 * d_udz,
	__global real * d_Src,
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

    real sigma[4][4];

    real4 udx = d_udx[I];
    real4 udy = d_udy[I];
    real4 udz = d_udz[I];

    // dalpha_u = (partial_t, partial_x, partial_y, partial_z)ux
    real4 dalpha_u[4] = {(real4)(udt.s0, udx.s0, udy.s0, udz.s0),
                         (real4)(udt.s1, udx.s1, udy.s1, udz.s1),
                         (real4)(udt.s2, udx.s2, udy.s2, udz.s2),
                         (real4)(udt.s3, udx.s3, udy.s3, udz.s3)};

    real DU[4] = {dot(u_new, dalpha_u[0]), dot(u_new, dalpha_u[1]),
                  dot(u_new, dalpha_u[2]), dot(u_new, dalpha_u[3])};
    real u[4] = {u_new.s0, u_new.s1, u_new.s2, u_new.s3};
    
    // theta = dtut + dxux + dyuy + dzuz where d=coviariant differential
    real theta = udt.s0 + udx.s1 + udy.s2 + udz.s3;

    real etav = ETAOS * S(e_v2.s0, eos_table) * hbarc;

    real pi1[10];
    for ( int mn=0; mn < 10; mn ++ ) {
        pi1[mn] = d_pi1[10*I+mn];
    }

    for ( int mu = 0; mu < 4; mu ++ )
        for ( int nu = mu; nu < 4; nu ++ ) {
            sigma[mu][nu] = dot(gm[mu], dalpha_u[nu]) + 
                            dot(gm[nu], dalpha_u[mu]) -
                            (u[mu]*DU[nu] + u[nu]*DU[mu]) -
                            2.0f/3.0f*(gmn[mu][nu]-u[mu]*u[nu])*theta;

            // set the sigma^{mu nu} and theta 0 when ed is too small
            if ( u[0] > 10 ) {
                sigma[mu][nu] = 0.0f;
                theta = 0.0f;
            }

            int mn = idx(mu, nu);
            real pi_old = pi1[mn] * u_old.s0;

            // /** step==1: Q' = Q0 + Src*DT
            //     step==2: Q  = Q0 + (Src(Q0)+Src(Q'))*DT/2
            // */

            real stiff_term = -(pi1[mn]-etav*sigma[mu][nu])/0.3f;
            real src = stiff_term - pi1[mn]*theta/3.0f - pi1[mn]*u_new.s0/tau;
            src -= (u[mu]*pi1[idx(nu,0)] + u[nu]*pi1[idx(mu,0)])*DU[0];
            src -= (u[mu]*pi1[idx(nu,1)] + u[nu]*pi1[idx(mu,1)])*DU[1];
            src -= (u[mu]*pi1[idx(nu,2)] + u[nu]*pi1[idx(mu,2)])*DU[2];
            src -= (u[mu]*pi1[idx(nu,3)] + u[nu]*pi1[idx(mu,3)])*DU[3];

            d_Src[idn(I, mn)] += src;
            pi_old = pi_old + d_Src[idn(I, mn)]*DT/step;

            d_pinew[10*I + mn] = pi_old/u_new.s0;
    }

    // u[0]*sigma[0][0]-u[1]*sigma[1][0]-u[2]*sigma[2][0]-u[3]*sigma[3][0],
    //d_checkpi[I] = (real4)(sigma[0][0]-sigma[1][1]-sigma[2][2]-sigma[3][3],
    //d_checkpi[I] = (real4)(udx.s0,
    d_checkpi[I] = (real4)(d_pinew[10*I + idx(0,1)],
    //d_checkpi[I] = (real4)(d_pinew[10*I]-d_pinew[10*I+idx(1,1)]-
    //                        d_pinew[10*I+idx(2,2)]-d_pinew[10*I+idx(3,3)],
    u[0]*sigma[0][1]-u[1]*sigma[1][1]-u[2]*sigma[2][1]-u[3]*sigma[3][1],
    u[0]*sigma[0][2]-u[1]*sigma[1][2]-u[2]*sigma[2][2]-u[3]*sigma[3][2],
    u[0]*sigma[0][3]-u[1]*sigma[1][3]-u[2]*sigma[2][3]-u[3]*sigma[3][3]);
    
}
