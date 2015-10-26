#include<helper.h>

real Nw(real Ta, real Tb);
real Nb(real Ta, real Tb);
real ed_transverse(real Ta, real Tb);
real weight_along_eta(real z, real etas0_);
real thickness(real x, real y);

// needs Ro0, R, Eta from gpu_define
real thickness(real x, real y)
{
	real r,thickness,f,z ,cut,a[10],b[10];
	real zk[5]={-0.9061798f, -0.5386492f, 0.0f, 0.5386493f, 0.9061798f};//range(-1,1)
	real Ak[5]={0.2369269f,0.4786287f,0.5688889f,0.4786287f,0.2369269f};//weight
	thickness = 0.0f;
	cut  = 5.0f*R;
	for (int j=0; j!=10; ++j){
		a[j] = j*cut / 10.0f;
		b[j] = (j+1)*cut / 10.0f;
		for(int i=0; i!=5; ++i){
			z=(a[j]+b[j])/2.0f + (b[j]-a[j])/2.0f* zk[i];
			r=sqrt( x*x+y*y+z*z );
			f=Ro0/(1.0f+exp((r-R)/Eta));
			thickness= thickness +Ak[i]*f*(b[j]-a[j])/2.0f;
		}
	}
	return 2.0f*thickness;
}

// energy deposition  along eta, define BJORKEN_SCALING to set heta=1
real weight_along_eta(real z, real etas0_) {
	real heta;
#ifdef BJORKEN_SCALING
    heta = 1.0f;
#else
	if( fabs(z-etas0_) > Eta_flat ) {
        heta=exp(-pow(fabs(z-etas0_)-Eta_flat,2.0f)/(2.0f*Eta_gw*Eta_gw));
    } else {
        heta = 1.0f;
    }
#endif 
	return heta;
}

// energy deposition in transverse plane from 2 components model
// (knw*nw(x,y) + knb*nb(x,y))*heta, b is impact parameter
real ed_transverse(real Ta, real Tb) {
    return (Hwn*Nw(Ta, Tb) + (1.0f-Hwn)*Nb(Ta, Tb));
}


// the rapidity twist due to forward-backward assymetry
inline real etas0(real Ta, real Tb) {
    // gamma_n = sqrt_s / (2*m_nucleon)
    double gamma_n = SQRTS / (2.0 * 0.938);
    double v_n = sqrt(1.0 - 1.0/(gamma_n*gamma_n));
    real T1 = (Ta + Tb)*gamma_n;
    real T2 = (Ta - Tb)*gamma_n*v_n;
    return 0.5f*log((T1+T2)/(T1-T2));
}

// number of wounded nucleons
real Nw(real Ta, real Tb) {
	return Ta*(1.0f-pow(1.0f-Si0*Tb/NumOfNucleons, NumOfNucleons))
		  +Tb*(1.0f-pow(1.0f-Si0*Ta/NumOfNucleons, NumOfNucleons));
}

// number of binary collisions
real Nb(real Ta, real Tb) {
	return Si0*Ta*Tb;
}


// each thread calc one slice of initial energy density with fixed
// (x, y) and varying eta_s
__kernel void glauber_ini( __global real4 * d_ev1 )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    real x = (i - NX/2)*DX;
    real y = (j - NY/2)*DY;
    real Tcent = thickness(0.0f, 0.0f);
    real ed_central = ed_transverse(Tcent, Tcent);

    real kFactor = Edmax / ed_central;
    real b = ImpactParameter;
    real Ta = thickness(x-0.5*b, y);
    real Tb = thickness(x+0.5*b, y);
    real edxy = kFactor*ed_transverse(Ta, Tb);
    real etas0_ = etas0(Ta, Tb);
    for ( int k = 0; k < NZ; k++ ) {
        real etas = (k - NZ/2)*DZ;
        real heta = weight_along_eta(etas, etas0_);
        d_ev1[i*NY*NZ + j*NZ + k] = (real4)(edxy*heta, 0.0f, 0.0f, 0.0f);
    }
}
