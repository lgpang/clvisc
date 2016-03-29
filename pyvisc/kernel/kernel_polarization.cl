# include <real_type.h>
# define mass 1.115f
# define beta (1.0f/0.137f)

// calc the Lambda polarization on freeze out hyper surface
// Each local work group is used to calculate polarization for one rapidity
__kernel void polarization_on_sf(
        __global real4 * d_polarization,
        __global real * d_density,
        __global const real8 * d_sf,
        __global const real * d_pimn,
        __global const real * d_omega_sf,
        __constant real * rapidity,
        __constant real * px,
        __constant real * py,
        const int size_sf) 
{
    __local real4 subpol[256];
    __local real  subspec[256];

    int lid = get_local_id(0);

    real8 sf;
    real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;

    // notice there if size_sf is not multiplies of 256, some threads can never
    // hit the barrier(CLK_LOCAL_MEM_FENCE), the program will abort
    for ( int i = get_local_id(0); i < 256000; i += get_local_size(0) ) {
        sf = d_sf[i];
        omega_tx = d_omega_sf[6*i + 0];
        omega_ty = d_omega_sf[6*i + 1];
        omega_tz = d_omega_sf[6*i + 2];
        omega_xy = d_omega_sf[6*i + 3];
        omega_xz = d_omega_sf[6*i + 4];
        omega_yz = d_omega_sf[6*i + 5];

        real4 umu = (real4)(1.0f, sf.s4, sf.s5, sf.s6) * \
            1.0f/sqrt(max((real)1.0E-15f, \
            (real)(1.0f-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        
        real4 dsigma = sf.s0123;
        real etas = sf.s7;

        for ( int ix = 0; ix < NPX; ix++ ) {
            for ( int iy = 0; iy < NPY; iy++ ) {
                real pt = sqrt(px[ix]*px[ix] + py[iy]*py[iy]);
                real mt = sqrt(mass*mass + pt*pt);
                for ( int ih = 0; ih < NRAPIDITY; ih++ ) {
                    real Y = rapidity[ih];
                    real mtcosh = mt * cosh(Y - etas);
                    real mtsinh = mt * sinh(Y - etas);
                    real4 pmu = (real4)(mtcosh, -px[ix], -py[iy], -mtsinh);

                    real pdotu = dot(umu, pmu);
                    real tmp = exp(-pdotu*beta);
                    real feq = tmp/(1.0f + tmp);
                    real volum = dot(dsigma, pmu);

                    //subspec[i] = volum * feq;
                    subspec[i] = 1.0E-5f;

                    // pomega_a = Omega^{a b} p_b
                    real pomega_0 =  omega_tx * pmu.s1 + omega_ty * pmu.s2 + omega_tz * pmu.s3;
                    real pomega_1 = -omega_tx * pmu.s0 + omega_xy * pmu.s2 + omega_xz * pmu.s3;
                    real pomega_2 = -omega_ty * pmu.s0 - omega_xy * pmu.s1 + omega_yz * pmu.s3;
                    real pomega_3 = -omega_tz * pmu.s0 - omega_xz * pmu.s1 - omega_yz * pmu.s2;

                    real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);

                    //subpol[i] = volum * pomega * feq * (1.0f - feq);
                    subpol[i] = (real4)(1.0E-5f, 1.0E-5f, 1.0E-5f, 1.0E-5f);

                    // notice that the last group of data
                    barrier(CLK_LOCAL_MEM_FENCE);

                    for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
                        if ( get_local_id(0) < s ) {
                            subspec[lid] += subspec[lid + s];
                            subpol[lid] += subpol[lid + s];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    
                    if ( lid == 0 ) {
                        d_density[ih*NPX*NPY + ix*NPY + iy] += subspec[0];
                        d_polarization[ih*NPX*NPY + ix*NPY + iy] += subpol[0];
                    }
        }}}
    }
}
