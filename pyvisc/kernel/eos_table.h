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

    real4 eos_low = read_imagef(eos_table, sampler, (int2)(col, row));
    // normal case, do interpolation between col and (col+1)
    col = col + 1;
    // when col is the last one in one row, row+=1
    if ( col == EOS_NUM_OF_COLS ) {
        row += 1;
        col = 0;
    }
    real4 eos_high = read_imagef(eos_table, sampler, (int2)(col, row));
    real r = (eps - eos_low.s0)/EOS_ED_STEP;
    
    // eos.s0123 = (ed, pr, T, entropy density)
    real4 epTs = (1.0f - r)*eos_low + r*eos_high;

    return epTs;
}

// get the pressure from eos_table
inline real P(real eps, read_only image2d_t eos_table){
    return eos(eps, eos_table).s1;
}

// get the pressure from eos_table
inline real T(real eps, read_only image2d_t eos_table){
    return eos(eps, eos_table).s2;
}

// get the pressure from eos_table
inline real S(real eps, read_only image2d_t eos_table){
    return eos(eps, eos_table).s3;
    //real4 epTs = eos(eps, eos_table);
    //return (epTs.s0 + epTs.s1)/max(1.0E-6f, epTs.s2);
}


#endif


#ifdef EOS_BINARY_SEARCH

constant const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE
          | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// read e, p, T, s from eos_table
inline real4 eos(real eps, read_only image2d_t eos_table){
    int row, col;
    real ed_step;
    if ( eps < 1.0E-5f ) {
        row = 0;
        ed_step = 1.0E-8f;
        col = floor(eps/ed_step);
    } else if ( eps < 1.0E-2f ) {
        row = 1;
        ed_step = 1.0E-5f;
        col = floor((eps - 1.0E-5f)/ed_step);
    } else if ( eps < 1.0E1f ) {
        row = 2;
        ed_step = 1.0E-2f;
        col = floor((eps - 1.0E-2f)/ed_step);
    } else if ( eps < 1.0E4f ) {
        row = 3;
        ed_step = 10.0f;
        col = floor((eps - 10)/ed_step);
    }

    real4 eos_low = read_imagef(eos_table, sampler, (int2)(col, row));
    real4 eos_high = read_imagef(eos_table, sampler, (int2)(col+1, row));
    real r = (eps - eos_low.s0)/ed_step;
    // eos.s0123 = (ed, pr, T, entropy density)
    real4 epTs = (1.0f - r)*eos_low + r*eos_high;

    return epTs;
}

// get the pressure from eos_table
inline real P(real eps, read_only image2d_t eos_table){
    return eos(eps, eos_table).s1;
}

// get the pressure from eos_table
inline real T(real eps, read_only image2d_t eos_table){
    return eos(eps, eos_table).s2;
}

// get the pressure from eos_table
inline real S(real eps, read_only image2d_t eos_table){
    return eos(eps, eos_table).s3;
    //real4 epTs = eos(eps, eos_table);
    //return (epTs.s0 + epTs.s1)/max(1.0E-6f, epTs.s2);
}


#endif



#endif

