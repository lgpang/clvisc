#ifndef __REAL_TYPE__
#define __REAL_TYPE__

#ifdef USE_SINGLE_PRECISION
typedef float  real;
typedef float4 real4;
typedef float8 real8;
typedef float3 real3;
#define acu 1.0E-14f
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                  
typedef double  real;
typedef double4 real4;
typedef double8 real8;
typedef double3 real3;
#define acu 1.0E-14
#endif


#endif
