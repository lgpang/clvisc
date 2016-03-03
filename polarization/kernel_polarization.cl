# define mass 1.115f
# define beta (1.0f/0.137f)

__kernel void polarization_on_sf(
        __global float * d_polarization,
        __global float * d_vorticity,
        __global float * d_density,
        __global const float4 * d_s,
        __global const float4 * d_u,
        __global const float * d_omegaY,
        __global const float * d_etas,
        const float rapidity,
        const float px,
        const float py,
        const int size_sf) {
    int i = get_global_id(0);
    if ( i < size_sf ) {
        float4 dsmu = d_s[i];
        float4 umu = d_u[i];
        float omega_y = d_omegaY[i];
        float etas = d_etas[i];

        float pt = sqrt(px*px + py*py);
        float mt = sqrt(mass*mass + pt*pt);
        float4 p_mu = (float4)(mt * cosh(rapidity - etas),
                -px, -py, -mt * sinh(rapidity - etas));
        float pdotu = dot(umu, p_mu);

        float volum = dot(dsmu, p_mu);

        float tmp = exp(-pdotu*beta);
        float nf = tmp/(1.0f + tmp);

        float pbar_sqr = mass*mass - pdotu*pdotu;

        float mass_fkt = 1.0f - pbar_sqr/(3*mass*(mass+pdotu));

        d_polarization[i] = -(volum*beta*omega_y*mass_fkt*pbar_sqr/(pdotu*pdotu)*nf*(1-nf))/6.0f;

        d_vorticity[i] = volum*omega_y*nf;

        d_density[i] = volum * nf;

        // why d_polarization and d_vorticity are not correct? compared to the python version.
    }
}
