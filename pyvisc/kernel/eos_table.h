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

inline real P(real eps, read_only image2d_t eos_table){
     return eps/3.0f;
}

inline real T(real eps, read_only image2d_t eos_table){
     return  hbarc1*pow( (real)1.0f/(dof*coef)*eps/hbarc1, (real)0.25f);
}

inline real S(real eps, read_only image2d_t eos_table){
     return  ( eps + P(eps, eos_table)) / fmax((real)1.0E-10f, T(eps, eos_table));
}

inline real CS2(real eps, read_only image2d_t eos_table){
     return  0.33333333f;
}

#endif

#ifdef EOS_TABLE

constant const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE
          | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// read e, p, T, s from eos_table
inline real4 eos(real eps, read_only image2d_t eos_table){
    real ed_per_row = EOS_ED_STEP*EOS_NUM_OF_COLS;
    int row = eps/ed_per_row;
    int col = (eps - EOS_ED_START - row*ed_per_row)
               /EOS_ED_STEP;

    real eps_low = EOS_ED_START + row * ed_per_row + col * EOS_ED_STEP;

    real4 eos_low = read_imagef(eos_table, sampler, (int2)(col, row));
    // normal case, do interpolation between col and (col+1)
    col = col + 1;
    // when col is the last one in one row, row+=1
    if ( col == EOS_NUM_OF_COLS ) {
        row += 1;
        col = 0;
    }
    real4 eos_high = read_imagef(eos_table, sampler, (int2)(col, row));
    real r = (eps - eps_low)/EOS_ED_STEP;
    
    // eos.s0123 = (cs2, pr, T, entropy density)
    real4 cpTs = (1.0f - r)*eos_low + r*eos_high;

    return cpTs;
}

// get the pressure from eos_table
inline real P(real eps, read_only image2d_t eos_table){
    if ( eps < EOS_ED_START + EOS_NUM_ED * EOS_ED_STEP ) {
        return eos(eps, eos_table).s1;
    } else { // the edmax from s9p5-pce-v1 is too small (~ 300 GeV/fm^3)
       return 0.3327f * eps - 0.3223f*pow(eps, (real)0.4585f) - 0.003906f*eps*exp(-0.05697f*eps);
    }
}

// get the pressure from eos_table
inline real S(real eps, read_only image2d_t eos_table){
    if ( eps < EOS_ED_START + EOS_NUM_ED * EOS_ED_STEP ) {
        return eos(eps, eos_table).s3;
    } else {
        return  18.202f*eps - 63.0218f - 4.85479f * exp(-2.72407E-11f*pow(eps,(real)4.54886f))
                + 65.1272f * pow(eps, -(real)0.128012f) *exp(-0.00369624f*pow(eps, (real)1.18735f));
    }
}


// get the pressure from eos_table
inline real T(real eps, read_only image2d_t eos_table){
    if ( eps < EOS_ED_START + EOS_NUM_ED * EOS_ED_STEP ) {
        return eos(eps, eos_table).s2;
    } else {
        return (eps + P(eps, eos_table)) / S(eps, eos_table);
    }
}

// get the speed of sound square
inline real CS2(real eps, read_only image2d_t eos_table){
    if ( eps < EOS_ED_START + EOS_NUM_ED * EOS_ED_STEP ) {
        return eos(eps, eos_table).s0;
    } else {
        return 0.333333333f;
    }
}



#endif




#endif

