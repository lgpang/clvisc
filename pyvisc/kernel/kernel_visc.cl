#include<helper.h>

/* d_SrcT: Src for viscous hydro dT^{tau mu}/dt terms
   d_SrcN: Src for conserved charge densities
   d_ev:   (ed, vx, vy, veta)
   d_nc:   (nb, ne, ns, null)
   d_pi:   (10 for pi^{mu nu} and 1 for PI)
*/
__kernel void kt_src_christoffel(
             __global real4 * d_SrcT,
             __global real4 * d_SrcN,
		     __global real4 * d_ev,
		     __global real4 * d_n,
		     __global real * d_pi,
             read_only image2d_t eos_table,
		     const real tau,
             const int step) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
        if ( step == 1 ) {
            d_Src[I] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
        real4 e_v = d_ev[I];
        real ed = e_v.s0;
        real vx = e_v.s1;
        real vy = e_v.s2;
        real vz = e_v.s3;
        real u0 = gamma(vx, vy, vz);

        real bulkpi = d_pi[idn(I, IDX_PI)];
        real pressure = P(ed, eos_table) + bulkpi;

        // Tzz_tilde = T^{eta eta} * tau^2; no 1/tau in vz
        real Tzz_tilde = (ed + pressure)*u0*u0*vz*vz + pressure
                         + d_pi[idn(I, idx(3, 3))];
        real Ttz_tilde = (ed + pressure)*u0*u0*vz
                         + d_pi[idn(I, idx(0, 3))];
        d_Src[I] = d_Src[I] - (real4)(Tzz_tilde, 0.0f, 0.0f, Ttz_tilde);
    }
}


// kt1d_visc to calc flux residual for T^mn=T_0^{mn} + pi^{mn}
// Qpi_ph = 0.5*(pi^{t mu}_ip1 + pi^{t mu}_i)
// Fpi_ph = 0.5*(pi^{w mu}_ip1 + pi^{w mu}_i) where w=x, y, eta

real4 kt1d_visc(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1,
                real4 ev_ip2, real4 Qpi_ph, real4 Qpi_mh,
                real4 Fpi_ph, real4 Fpi_mh,
           real tau, int along, read_only image2d_t eos_table);

real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1,
           real4 ev_ip2, real4 Q_pi, real4 Flux_pi, real tau,
           int along, read_only image2d_t eos_table) {
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
