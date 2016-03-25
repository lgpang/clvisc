#include<helper.h>

// invariant vorticity vector 
// omega^{mu nu} = epsilon^{mu nu a b} d_a u_b
// omega^{tau x} = dyuz - dzuy
// omega^{tau y} = -(dxuz - dzux)
// omega^{tau z} = dxuy - dyux
// omega^{x y} = dtuz - dzut
// omega^{x z} = -(dtuy - dyut)
// omega^{y z} = dtux - dxut

// Covariant derivatives?

// calc beta*umu from (ed, vx, vy, tau^2*veta) float4 vector
inline real4 ubeta(real4 ev, read_only image2d_t eos_table)
{
    return umu4(ev)/T(ev.s0, eos_table);
}

// wrapper for address index
inline int address(int i, int j, int k)
{
    return i*NY*NZ + j*NZ + k;
}

__kernel void omega(
    __global real4 * d_ev1,
    __global real4 * d_ev2,
	__global real  * d_omega,
    read_only image2d_t eos_table,
	const real tau)
{
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    real4 uold = ubeta(d_ev1[address(I, J, K)], eos_table);
    real4 unew = ubeta(d_ev2[address(I, J, K)], eos_table);

    real4 dudt = (unew - uold)/DT;
    real4 dudx = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    if ( I != 0 && I != NX-1 ) {
        dudx = (ubeta(d_ev2[address(I+1, J, K)], eos_table)
              - ubeta(d_ev2[address(I-1, J, K)], eos_table)) / (2.0f*DX);
    } else if ( I == 0 ) { 
        dudx = (ubeta(d_ev2[address(I+1, J, K)], eos_table) - unew) / DX;
    } else if ( I == NX-1 ) {
        dudx = (unew - ubeta(d_ev2[address(I-1, J, K)], eos_table)) / DX;
    }

    real4 dudy = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    if ( J != 0 && J != NY-1 ) {
        dudy = (ubeta(d_ev2[address(I, J+1, K)], eos_table)
              - ubeta(d_ev2[address(I, J-1, K)], eos_table)) / (2.0f*DY);
    } else if ( J == 0 ) { 
        dudy = (ubeta(d_ev2[address(I, J+1, K)], eos_table) - unew) / DY;
    } else if ( J == NY-1 ) {
        dudy = (unew - ubeta(d_ev2[address(I, J-1, K)], eos_table)) / DY;
    }

    // initialize with Christoffel symbols
    real4 dudz = (real4)(unew.s3, 0.0f, 0.0f, unew.s0)/tau;
    if ( K != 0 && K != NZ-1 ) {
        dudz += (ubeta(d_ev2[address(I, J, K+1)], eos_table)
              - ubeta(d_ev2[address(I, J, K-1)], eos_table)) / (2.0f*DZ*tau);
    } else if ( K == 0 ) { 
        dudz += (ubeta(d_ev2[address(I, J, K+1)], eos_table) - unew) / (DZ*tau);
    } else if ( K == NZ-1 ) {
        dudz += (unew - ubeta(d_ev2[address(I, J, K-1)], eos_table)) / (DZ*tau);
    }

    d_omega[6*address(I,J,K)+0] = dudy.s3 - dudz.s2;
    d_omega[6*address(I,J,K)+1] = -(dudx.s3 - dudz.s1);
    d_omega[6*address(I,J,K)+2] = dudx.s2 - dudy.s1;
    d_omega[6*address(I,J,K)+3] = dudt.s3 - dudz.s0;
    d_omega[6*address(I,J,K)+4] = -(dudt.s2 - dudy.s0);
    d_omega[6*address(I,J,K)+5] = dudt.s1 - dudx.s0;
}
