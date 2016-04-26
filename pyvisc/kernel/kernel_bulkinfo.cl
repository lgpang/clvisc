#include<helper.h>

// used for parallel reduction in momentum eccentricity
#define block_size 256

// get the round_up value 
int round_up(int value, int multiple) {
    int rem = value % multiple;
    if ( rem != 0 ) {
        value += multiple - rem;
    }
    return value;
}


// d_ed_lab is the energy density at lab frame
// boosted from Bjorken frame with y = space rapidity etas
__kernel void total_energy_and_entropy(
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


// calc the momentum eccentricity eccp1 and eccp2 as a function of
// rapidity

__kernel void eccp_vs_rapidity(
             __global real * d_eccp1,
             __global real * d_eccp2,
		     __global real4 * d_ev,
             read_only image2d_t eos_table,
		     const real tau) {
    int k = get_group_id(0);

    __local real Txx[block_size];
    __local real Tyy[block_size];
    __local real T0x[block_size];

    real Txx_sum = 0.0f;
    real Tyy_sum = 0.0f;
    real T0x_sum = 0.0f;

    // notice there if size_cells is not multiplies of BSZ,
    // some threads can never hit the barrier(CLK_LOCAL_MEM_FENCE),
    // the program will abort
    int ncells = round_up(NX*NY, block_size);

    int lid = get_local_id(0);
    for ( int id = lid; id < ncells; id += block_size ) {
        int i = id / NY;
        int idx = id * NZ + k;
        real4 e_v = d_ev[idx];
        real ed = e_v.s0;
        real vx = e_v.s1;
        real vy = e_v.s2;
        real vz = e_v.s3;
        real pr = P(ed, eos_table);
        real u0 = gamma(vx, vy, vz);

        if ( id < NX * NY ) {
            Txx[lid] = (ed + pr)*u0*u0*vx*vx + pr;
            Tyy[lid] = (ed + pr)*u0*u0*vy*vy + pr;
            T0x[lid] = (ed + pr)*u0*u0*vx;
        } else {
            Txx[lid] = 0.0f;
            Tyy[lid] = 0.0f;
            T0x[lid] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // parallel reduction 
        for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
            if ( lid < s ) {
                Txx[lid] += Txx[lid + s];
                Tyy[lid] += Tyy[lid + s];
                T0x[lid] += T0x[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if ( lid == 0 ) {
             Txx_sum += Txx[0];
             Tyy_sum += Tyy[0];
             T0x_sum += T0x[0];
        }
    }

    if ( lid == 0 ) {
        d_eccp1[k] = T0x_sum / (Txx_sum + Tyy_sum);
        d_eccp2[k] = (Txx_sum - Tyy_sum) / (Txx_sum + Tyy_sum);
    }
}
