#ifndef __HELPER_H__
#define __HELPER_H__
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

#pragma OPENCL EXTENSION cl_amd_printf: enable

//#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

#ifdef USE_SINGLE_PRECISION
typedef float  real;
typedef float2  real2;
typedef float3 real3;
typedef float4 real4;
typedef float8 real8;
typedef float16 real16;
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                  
typedef double  real;
typedef double2  real2;
typedef double3 real3;
typedef double4 real4;
typedef double8 real8;
typedef double16 real16;
#endif



#endif
