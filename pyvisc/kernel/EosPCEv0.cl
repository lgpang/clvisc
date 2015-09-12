#ifndef __EOS_H__
#define __EOS_H__

#include "real_type.h"

/** \breif EOS EOSI, s95p-PCE165-v0 from TECHQM */
/** Pressure as a function of energy density in units GeV/fm^3 */
#ifdef EOSI

#define  dof  (169.0f/4.0f)
#define  hbarc1  0.1973269f
#define  hbarc3  pow(0.1973269631f, 3.0f)
#define  coef  (M_PI_F*M_PI_F/30.0f)

inline real P(real eps){
     return eps/3.0f;
}

inline real T( real eps ){
     return  hbarc1*pow( (real)1.0f/(dof*coef)*eps/hbarc1, (real)0.25f);
}

inline real S( real eps ){
     return  ( eps + P(eps) ) / fmax((real)1.0E-10f, T(eps));
}

#endif

#ifdef EOSLPCE

inline real P( real eps ){
    if (eps < 0.5028563305441270 ) 
       return 0.3299 * (exp( 0.4346*eps ) - 1.0);
    else if( eps < 1.62 )
       return 1.024E-7*exp( 6.041*eps )+0.007273 + 0.14578*eps;
    else if( eps < 1.86 ) /**< \bug presure gradient has a ripple here */
       return 0.30195*exp( 0.31308*eps ) - 0.256232 ;
    else if(eps < 9.9878 )
       return 0.332*eps - 0.3223*pow(eps, (real)0.4585) - 0.003906 *eps*exp(-0.05697*eps)\
              +0.1167*pow(eps, (real)(-1.233)) + 0.1436*eps*exp( - 0.9131 *eps );
    else
       return 0.3327 * eps - 0.3223*pow( eps, (real)0.4585) - 0.003906*eps*exp(-0.05697*eps);
}

inline real S( real eps ){
  real Sp;
  if ( eps < 0.1270769021427449 )
  Sp = 12.2304 * pow( eps, (real)1.16849 );  
  else if( eps < 0.446707952467404 )
  Sp = 11.9279 * pow( eps, (real)1.15635 );
  else if( eps < 1.9402832534193788 )
  Sp = 0.0580578 + 11.833 * pow( eps, (real)1.16187 );
  else if( eps < 3.7292474570977285 )
  Sp = 18.202 * eps - 62.021814 - 4.85479 * exp( -2.72407E-11 * pow(eps,(real)4.54886)) + 65.1272 * pow( eps, (real)(-0.128012))*exp( -0.00369624 * pow( eps, (real)1.18735 )) - 4.75253*pow( eps, -(real)1.18423 );
  else Sp = 18.202*eps - 63.0218 - 4.85479 * exp( -2.72407E-11*pow(eps,(real)4.54886)) + 65.1272 * pow( eps, -(real)0.128012 ) *exp( -0.00369624*pow(eps, (real)1.18735));

  return pow( Sp, (real)0.75 );
}

inline real T( real eps ){
   if ( eps < 0.51439398 ) return 0.203054 * pow(eps, (real)0.30679);
   else return ( eps + P(eps) )/S(eps) ;
}


#endif



#ifdef EOSLCE
#include<eos_spline.h>

////current EOSLCE only works for ed<311.001

inline real fspline(real eps, constant real4 * spline ){
    int index = 0;
    real xlow = 0.0;
    if( eps > 0.0 && eps < 0.001 ){
        index = 0;
        xlow = 0.0;
    }
    else if ( eps < 1.001 ) {
        index = 1 + (int)( (eps-0.001)/0.1 );
        xlow = 0.001 + (index-1)*0.1;
    }
    else if ( eps < 11.001 ) {
        index = 11 + (int)( (eps-1.001)/1.0 );
        xlow = 1.001 + (index-11)*1.0;
    }
    else if ( eps < 61.001 ) {
        index = 21 + (int)( (eps-11.001)/5.0 );
        xlow = 11.001 + (index-21)*5.0;
    }
    else if ( eps < 311.001 ) {
        index = 31 + (int)( (eps-61.001)/25.0 );
        xlow = 61.001 + (index-31)*25.0;
    }

    real4 s = spline[index];
    return s.s0 + (eps-xlow)*(s.s1 + (eps-xlow)*(s.s2+(eps-xlow)*s.s3));
}

inline real P(real eps){
    return  fspline(eps, spline_p);
}

inline real T(real eps){
    return  fspline(eps, spline_T);
}

inline real S(real eps){
    return  fspline(eps, spline_s);
}

#endif

#endif

