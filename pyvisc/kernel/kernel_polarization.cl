# define mass 1.115f
# define beta (1.0f/0.137f)

// calc the Lambda polarization on freeze out hyper surface
// Each local work group is used to calculate polarization for one rapidity
__kernel void polarization_on_sf(
        __global float4 * d_polarization,
        __global float * d_density,
        __global const float8 * d_sf,
        __global const float * d_pimn,
        __global const float * d_omega_sf,
        __constant float * rapidity,
        __constant float * px,
        __constant float * py,
        const int size_sf) 
{
    __local real4 subpol[256];
    __local real  subspec[256];

    real8 sf;
    real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;

    for ( int i = get_local_id(0); i < size_sf; i += get_local_size(0) ) {
        sf = d_sf[i];
        omega_tx = d_omega_sf[6*i + 0];
        omega_ty = d_omega_sf[6*i + 1];
        omega_tz = d_omega_sf[6*i + 2];
        omega_xy = d_omega_sf[6*i + 3];
        omega_xz = d_omega_sf[6*i + 4];
        omega_yz = d_omega_sf[6*i + 5];

        real4 umu = (real4)(1.0f, sf.s4, sf.s5, sf.s6) * \
            1.0f/sqrt(max((real)1.0E-15, \
            (real)(1.0-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        
        real4 dsigma = sf.s0123;
        real etas = sf.s7;

        for ( int ix = 0; ix < NPX; ix++ ) {
            for ( int iy = 0; iy < NPY; iy++ ) {
                real pt = sqrt(px*px + py*py);
                real mt = sqrt(mass*mass + pt*pt);
                for ( int ih = 0; ih < NRAPIDITY; ih++ ) {
                    real Y = rapidity[ih];
                    real mtcosh = mt*cosh(rapidity-etas);
                    real mtsinh = mt*sinh(rapidity-etas);
                    real4 pmu = (real4)(mtcosh, -px, -py, -mtsinh);

                    real pdotu = dot(umu, pmu);
                    real tmp = exp(-pdotu*beta);
                    real feq = tmp/(1.0f + tmp);
                    real volum = dot(dsigma, pmu);

                    real pbar_sqr = mass*mass - pdotu*pdotu;

                    subspec[i] = dof * volum * feq;

                    // pomega_a = Omega^{a b} p_b
                    real pomega_0 =  omega_tx * pmu.s1 + omega_ty * pmu.s2 + omega_tz * pmu.s3;
                    real pomega_1 = -omega_tx * pmu.s0 + omega_xy * pmu.s2 + omega_xz * pmu.s3;
                    real pomega_2 = -omega_ty * pmu.s0 - omega_xy * pmu.s1 + omega_yz * pmu.s3;
                    real pomega_3 = -omega_tz * pmu.s0 - omega_xz * pmu.s1 - omega_yz * pmu.s2;

                    real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);

                    subpol[i] = dof * volum * pomega * feq * (1.0f - feq);

                    barrier(CLK_LOCAL_MEM_FENCE);

                    for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
                        if ( i < s ) {
                            subspec[i] += subspec[i + s];
                            subpol[i] += subpol[i + s];
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    
                    if ( i == 0 ) {
                        d_density[ih*NPX*NPY + ix*NPY + iy] += subspec[0];
                        d_polarization[ih*NPX*NPY + ix*NPY + iy] += subspec[0];
                    }
        }}}
    }
}
