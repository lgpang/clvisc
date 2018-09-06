#include<helper.h>

__kernel void kernel_bulk3d(  
        __global float * d_bulk3d_1step,
        __global real4 * d_ev,
        __global real  * d_shear_pi,
        __global real  * d_bulk_pi,
        read_only image2d_t eos_table)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int ni = get_global_size(0);
    int nj = get_global_size(1);
    int nk = get_global_size(2);

    size_t src_idx = (i * NXSKIP) * NY * NZ + (j * NYSKIP) * NZ + k * NZSKIP;

    float4 edv = d_ev[src_idx];
    float4 cpTs = eos(edv.s0, eos_table);
    float pressure = cpTs.s1;
    float temperature = cpTs.s2;
    float entropy_density = cpTs.s3;
    /** qgp_fraction=0 for T<0.184 and qgp_fraction=1 for T>0.220 */
    float qgp_fraction = 1.0f;
    if (temperature < 0.184f) {
        qgp_fraction = 0.0f;
    } else {
        qgp_fraction = lin_int(0.184f, 0.220f, 0.0f, 1.0f, temperature);
    }
    size_t idx = i * nj * nk + j * nk + k;
    d_bulk3d_1step[idx * 18 + 0] = edv.s0;                           //energy density
    d_bulk3d_1step[idx * 18 + 1] = entropy_density;
    d_bulk3d_1step[idx * 18 + 2] = temperature;
    d_bulk3d_1step[idx * 18 + 3] = pressure;
    d_bulk3d_1step[idx * 18 + 4] = edv.s1;                           // vx
    d_bulk3d_1step[idx * 18 + 5] = edv.s2;                           // vy
    float etas = (k*NZSKIP - 0.5f*NZ) * DZ;
    float rapidity = atanh(edv.s3) + etas;
    d_bulk3d_1step[idx * 18 + 6] = tanh(rapidity);                   // vz
    d_bulk3d_1step[idx * 18 + 7] = qgp_fraction;
    d_bulk3d_1step[idx * 18 + 8] = d_shear_pi[10 * src_idx + 0];     //pi00
    d_bulk3d_1step[idx * 18 + 9] = d_shear_pi[10 * src_idx + 1];     //pi01
    d_bulk3d_1step[idx * 18 + 10] = d_shear_pi[10 * src_idx + 2];    //pi02
    d_bulk3d_1step[idx * 18 + 11] = d_shear_pi[10 * src_idx + 3];    //pi03
    d_bulk3d_1step[idx * 18 + 12] = d_shear_pi[10 * src_idx + 4];    //pi11
    d_bulk3d_1step[idx * 18 + 13] = d_shear_pi[10 * src_idx + 5];    //pi12
    d_bulk3d_1step[idx * 18 + 14] = d_shear_pi[10 * src_idx + 6];    //pi13
    d_bulk3d_1step[idx * 18 + 15] = d_shear_pi[10 * src_idx + 7];    //pi22
    d_bulk3d_1step[idx * 18 + 16] = d_shear_pi[10 * src_idx + 8];    //pi23
    d_bulk3d_1step[idx * 18 + 17] = d_shear_pi[10 * src_idx + 9];    //pi33
}
