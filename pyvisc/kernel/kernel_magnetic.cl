#include<helper.h>

inline real eB( real x, real y, real eta, real t ){
    real s_x = SIGX;
    real s_y = SIGY;
    real eB_0 = EB0;
    real t_d = TAUD;
    return eB_0 * exp(-y*y/(2*s_y*s_y))*exp(-x*x/(2*s_x*s_x))*exp(-t/t_d);
}


inline real forcedensity(real x, real y, real eta, real T, real time, char along_direction){
    real hbarc3 = 0.19732f*0.19732f*0.19732f;
    real chiT = 1.0f/(3.0f*M_PI_F*M_PI_F)*log(T/0.110f);

    // do cut off to make sure chi(T) is bigger than 0
    if ( T < 0.110f ) chiT = 0.0f;
    
    real eBxy = eB(x,y,eta,time);

    if ( along_direction == 'x' ){
       return chiT * eBxy * ( eB(x+0.02f,y,eta,time)-eB(x-0.02f,y,eta,time) )/0.04f/hbarc3;
    } else if ( along_direction == 'y' ){
       return chiT * eBxy * ( eB(x, y+0.02f,eta,time)-eB(x,y-0.02f,eta,time) )/0.04f/hbarc3;
    }
}



__kernel void kt_src_magnetic(
             __global real4 * d_Src,
		     __global real4 * d_ev,
             read_only image2d_t eos_table,
		     const real tau,
             const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    real x = (I-NX/2)*DX;
    real y = (J-NY/2)*DY;
    real z = (K-NZ/2)*DZ;
    int IND = I*NY*NZ + J*NZ + K;

    real4 edv = d_ev[IND];
    real temperature = T(edv.s0, eos_table);

    real fx = forcedensity(x, y, z, temperature, tau, 'x');
    real fy = forcedensity(x, y, z, temperature, tau, 'y');

    real px = 0.0f;
    if ( I != 0 && I != NX-1 ) {
        px = (P(d_ev[(I+1)*NY*NZ + J*NZ + K].s0, eos_table)
            -  P(d_ev[(I-1)*NY*NZ + J*NZ + K].s0, eos_table))/(2.0f*DX);
    } else if ( I == 0 ) {
        px = (P(d_ev[(I+1)*NY*NZ + J*NZ + K].s0, eos_table)
            -  P(d_ev[(I)*NY*NZ + J*NZ + K].s0, eos_table))/DX;
    } else if ( I == NX-1 ) {
        px = (P(d_ev[(I)*NY*NZ + J*NZ + K].s0, eos_table)
            -  P(d_ev[(I-1)*NY*NZ + J*NZ + K].s0, eos_table))/DX;
    }

    real py = 0.0f;
    if ( J != 0 && J != NY-1 ) {
        py = (P(d_ev[I*NY*NZ + (J+1)*NZ + K].s0, eos_table)
            -  P(d_ev[I*NY*NZ + (J-1)*NZ + K].s0, eos_table))/(2.0f*DY);
    } else if ( J == 0 ) {
        py = (P(d_ev[I*NY*NZ + (J+1)*NZ + K].s0, eos_table)
            -  P(d_ev[I*NY*NZ + (J)*NZ + K].s0, eos_table))/(DY);
    } else if ( J == NY-1 ) {
        py = (P(d_ev[I*NY*NZ + (J)*NZ + K].s0, eos_table)
            -  P(d_ev[I*NY*NZ + (J-1)*NZ + K].s0, eos_table))/(DY);
    }

    //real src_x = (fabs(px)>fabs(fx)) ? (tau * fx) : tau*px;
    //real src_y = (fabs(py)>fabs(fy)) ? (tau * fy) : tau*py;

    real src_x = tau*fx;
    real src_y = tau*fy;

    // set J^{0} = vx*J^{x} + vy*J^[y}
    //real src_t = edv.s1*src_x + edv.s2*src_y;
    real src_t = 0.0f;

    d_Src[IND] += (real4)(src_t, src_x, src_y, 0.0f);
}
