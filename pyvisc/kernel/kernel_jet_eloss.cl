#include<helper.h>

/* \nabla T^{\mu\nu} = J^{\nu}
where $J^{\nu} is the energy momentum deposition from jet
energy loss;
*/

__kernel void jet_eloss_src(
             __global real4 * d_SrcT,
		     __global real4 * d_ev,
		     const real tau,
             const real jet_direction,
             const int jet_start_pos_index,
             read_only image2d_t eos_table) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    real x = (I - NX/2) * DX;
    real y = (J - NY/2) * DY;
    real z = (K - NZ/2) * DZ;
    //real etas = (K - NZ/2) * DZ;
    //real z = tau * sinh(etas);
    real3 speed = {cos(jet_direction), sin(jet_direction), 0.0};

    int I0 = jet_start_pos_index / (NY * NZ);
    int J0 = (jet_start_pos_index - I0 * NY * NZ) / NZ;
    int K0 = jet_start_pos_index - I0 * NY * NZ - J0 * NZ;
    real3 position = (real3){(I0 - NX/2)*DX, (J0 - NY/2)*DY, (K0 - NZ/2)*DZ} + speed * (tau - TAU0);

    //real3 position = (real3){0.0, 0.0, 0.0} + speed * (tau - TAU0);
    // since our \tilde{T}^{mn} = tau * T^{mn} when m != eta, n!=eta
    // since our \tilde{T}^{mn} = tau * tau * T^{m\eta} when m != eta
    // since our \tilde{T}^{mn} = tau * tau * tau * T^{m\eta} when m = eta

    real4 coef = {tau, tau, tau, tau*tau};

    real sigma_x = 0.3;
    real sigma_y = 0.3;
    real sigma_z = 0.3;

    real norm = 1.0 / (pow(sqrt(2.0f*M_PI_F), 3.0f) * sigma_x * sigma_y * sigma_z * tau);

    coef *= norm * exp(-(x - position.s0)*(x - position.s0)/(2 * sigma_x * sigma_x)
                           -(y - position.s1)*(y - position.s1)/(2 * sigma_y * sigma_y)
                           -(z - position.s2)*(z - position.s2)/(2 * sigma_z * sigma_z));

    int loc_i = (int)(position.s0/DX + NX/2);
    int loc_j = (int)(position.s1/DY + NY/2);
    int loc_k = (int)(position.s2/DZ + NZ/2);

    if ( loc_i >= 0 && loc_i < NX &&
         loc_j >= 0 && loc_j < NY &&
         loc_k >= 0 && loc_k < NZ ) {
        int jet_current_pos_index = loc_i*NY*NZ + loc_j*NZ + loc_k;
        real4 ev = d_ev[jet_current_pos_index].s0;

        real local_T = T(ev.s0, eos_table);

        // 300 MeV for dEt/dx = 14.0 GeV/fm
        real local_dEdx = 14.0f * pow(local_T / 0.3f, 3.0f);

        real4 dEdX = (real4){local_dEdx,
                         local_dEdx * cos(jet_direction),
                         local_dEdx * sin(jet_direction),
                         0.0f};

        d_SrcT[I * NY * NZ + J * NZ  + K] = d_SrcT[I * NY * NZ + J * NZ  + K] + coef * dEdX;
    }
}
