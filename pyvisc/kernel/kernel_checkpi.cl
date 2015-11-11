#include<helper.h>


__kernel void pizz_o_ep(
             __global real * d_pizz_o_ep,
             __global real * d_pi1,
		     __global real4 * d_ev,
             read_only image2d_t eos_table) {
    int K = get_global_id(0);

    real ratio = 0.0f;
    int num_of_cells = 0;

    int start_idx = 9 * SIZE;
    for ( int i = 0; i < NX; i ++ )
    for ( int j = 0; j < NY; j ++ ) {
       int I = i*NY*NZ + j*NZ + K;
       real4 ev = d_ev[I];
       real ep = ev.s0 + P(ev.s0, eos_table);

       if ( ep > 0.133f ) {
           ratio += d_pi1[start_idx + I] / ep;
           num_of_cells ++;
       }
    }

    d_pizz_o_ep[K] = ratio/num_of_cells;
}
