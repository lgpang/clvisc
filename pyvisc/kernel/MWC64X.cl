//small and fast random number generator that can run on gpu

#include<skip_mwc.cl>
enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

/*! generate a Random number between [0,1) */
real MWC64X(uint2 *state)
{
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,MWC64X_A);       // Step the RNG
    x=x*MWC64X_A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);               // Pack the state back up
    return res / ((real) UINT_MAX);    // Return a float num between 0-1 
}

/*! Skip function used to get the next random seed with a skipping distance */
uint2 MWC64X_Skip(uint2 state, ulong distance)
{
	return MWC_SkipImpl_Mod64(state, MWC64X_A, MWC64X_M, distance);
}
