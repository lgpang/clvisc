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

constant const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE
          | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline real P(real eps, read_only image2d_t eos_table){
    real ed_per_row = EOS_ED_STEP*EOS_NUM_OF_COLS;
    int row = eps/ed_per_row;
    int col = (eps - EOS_ED_START - row*ed_per_row)
               /EOS_ED_STEP;

    real4 eos_low = read_imagef(eos_table, sampler, (int2)(col, row));
    // normal case, do interpolation between col and (col+1)
    col = col + 1;
    // when col is the last one in one row, row+=1
    if ( col == EOS_NUM_OF_COLS - 1 ) {
        row += 1;
        col = 0;
    }
    real4 eos_high = read_imagef(eos_table, sampler, (int2)(col, row));
    real r = (eps - eos_low.s0)/EOS_ED_STEP;
    
    // eos.s0123 = (ed, pr, T, entropy density)
    real4 eos = (1.0f - r)*eos_low + r*eos_high;

    return eos.s1;
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

