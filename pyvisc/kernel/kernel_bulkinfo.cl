#include<helper.h>

// d_ed_lab is the energy density at lab frame
// boosted from Bjorken frame with y = space rapidity etas
__kernel void total_energy(
             __global real * d_ed_lab,
             __global real * d_s_lab,
		     __global real4 * d_ev,
             read_only image2d_t eos_table,
		     const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    int idx = I * NY * NZ + J * NZ + K;

    real4 e_v = d_ev[idx];
    real ed = e_v.s0;
    real vx = e_v.s1;
    real vy = e_v.s2;
    real vz = e_v.s3;
    real pr = P(ed, eos_table);
    real u0 = gamma(vx, vy, vz);

    real etas = (K - NZ / 2) * DZ;

    real cosh_etas = cosh(etas);

    real ed_lab = (ed + pr) * u0 * u0 * (cosh_etas + vz * sinh(etas))
                    - pr * cosh_etas;
    d_ed_lab[idx] = ed_lab;

    d_s_lab[idx] = u0 * S(ed, eos_table);
}
