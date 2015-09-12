#ifndef __HELPERH__
#define __HELPERH__
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//#pragma OPENCL EXTENSION cl_amd_printf: enable

#include"real_type.h"
#include"EosPCEv0.cl"

#define theta ((real)1.1)
#define hbarc ((real)0.19733)
#define acu ((real)1.0E-8)

#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))

/** 1D linear interpolation */
inline real lin_int( real x1, real x2, real y1, real y2, real x )
{
    real r = ( x - x1 )/ (x2-x1 );
    return y1*(1.0-r) + y2*r ;
}

inline real poly3_int( real x1, real x2, real x3, real y1, real y2, real y3, real x )
{
  return y1*(x-x2)*(x-x3)/((x1-x2)*(x1-x3)) + y2*(x-x1)*(x-x3)/((x2-x1)*(x2-x3)) + y3*(x-x1)*(x-x2)/((x3-x1)*(x3-x2)) ;
}


/** Flux limit for scalar */
inline real minmod( real x, real y, real z )
{
    if( x>0 && y>0 && z>0 ) return min(min(x,y),z);
    else if( x<0 && y<0 && z<0 ) return max(max(x,y),z);
    else return 0.0;
}



/** \Notice vector compare return -1 for true, 0 for false 
 *  real4 in function parameter must be pointer type */
real4 minmod4( real4 * x, real4 * y, real4 * z )
{

    return (real4) ( minmod((*x).s0, (*y).s0, (*z).s0),
                     minmod((*x).s1, (*y).s1, (*z).s1),
                     minmod((*x).s2, (*y).s2, (*z).s2),
                     minmod((*x).s3, (*y).s3, (*z).s3) );
}

/** Calc maximum local propagation speed */
inline real maxPropagationSpeed( real ut, real uk, real cs2 ){
    real ut2 = ut*ut;
    real uk2 = uk*uk;
   return (fabs(ut*uk*(1.0-cs2)) \
           + sqrt((ut2-uk2-(ut2-uk2-1.0)*cs2)*cs2))\
       /(ut2 - (ut2-1.0)*cs2);
//    return fabs( uk/ut ) + sqrt(cs2);
}



/** solve energy density from T00 and K=sqrt(T01**2 + T02**2 + T03**2)
 * */

inline void rootFinding( real* EdFind, real * T00, real *K2 ){
    real E0, E1;
    E1 = *T00;   /*predict */
//    *EdFind = -(*T00) + sqrt( 4.0*(*T00)*(*T00) - 3.0*(*K2) ) ;
    int i = 0;
    while( true ){
        E0 = E1;
        E1 = *T00 - (*K2)/( *T00 + P(E1)) ; 
        i ++ ;
        //i>20 is stable
        if( i>15 || fabs(E1-E0)/max( fabs(E1), (real)acu)<acu )
            break;
    }

    * EdFind = E1;
}



/** 3D KT algorithm for Tm0 and pi^{mn} updating*/
real HX( local real AB [BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
         local real CS2[BSZ][BSZ][BSZ],  real tau, int i, int j, int k, int I, int J, int K ){
   real DA0, DA1;
   //// DAX(i, j, k)
//   if( I==0 ){ 
//     AB[1][j][k] = poly3_int( 2.0, 3.0, 4.0, AB[2][j][k], AB[3][j][k], AB[4][j][k], 1.0 );
//     AB[0][j][k] = poly3_int( 1.0, 2.0, 3.0, AB[1][j][k], AB[2][j][k], AB[3][j][k], 0.0 );
//   }
//   if( I==NX-1 ){
//     AB[BSZ-2][j][k] = poly3_int( BSZ-3.0,BSZ-4.0, BSZ-5.0, AB[BSZ-3][j][k], AB[BSZ-4][j][k], AB[BSZ-5][j][k], BSZ-2.0 );
//     AB[BSZ-1][j][k] = poly3_int( BSZ-2.0,BSZ-3.0, BSZ-4.0, AB[BSZ-2][j][k], AB[BSZ-3][j][k], AB[BSZ-4][j][k], BSZ-1.0 );
//   }

   DA0 = minmod( theta*(AB[i+1][j][k]-AB[i][j][k]), \
         0.5*( AB[i+1][j][k] - AB[i-1][j][k] ), \
         theta*(AB[i][j][k]-AB[i-1][j][k]) );

   //// DAX(i+1, j, k)
   DA1 = minmod( theta*(AB[i+2][j][k]-AB[i+1][j][k]), \
         0.5*( AB[i+2][j][k] - AB[i][j][k] ), \
         theta*(AB[i+1][j][k]-AB[i][j][k]) );

  
real lam0 = maxPropagationSpeed( Umu[i][j][k].s0, Umu[i][j][k].s1, CS2[i][j][k] ) ;
real lam1 = maxPropagationSpeed( Umu[i+1][j][k].s0, Umu[i+1][j][k].s1, CS2[i+1][j][k] ) ;
//   real lam = 0.5*(lam0+lam1);
   real lam = lam0;
//   real lam = max( lam0, lam1 );

   real  AL = AB[i][j][k]   + 0.5 * DA0;
   real  AR = AB[i+1][j][k] - 0.5 * DA1;

//   real Jp = lin_int( 0.0, 1.0, AB[i][j][k]*Umu[i][j][k].s1/Umu[i][j][k].s0, \
//                         AB[i+1][j][k]*Umu[i+1][j][k].s1/Umu[i+1][j][k].s0, \
//                         0.5 + DT/DX*lam );
//   real Jm = lin_int( 0.0, 1.0, AB[i][j][k]*Umu[i][j][k].s1/Umu[i][j][k].s0, \
//                         AB[i+1][j][k]*Umu[i+1][j][k].s1/Umu[i+1][j][k].s0, \
//                         0.5 - DT/DX*lam );

   real Jp = AR * lin_int( 0.0, 1.0, Umu[i][j][k].s1/Umu[i][j][k].s0, \
                         Umu[i+1][j][k].s1/Umu[i+1][j][k].s0, \
                         0.5 );
   real Jm = AL * lin_int( 0.0, 1.0, Umu[i][j][k].s1/Umu[i][j][k].s0, \
                         Umu[i+1][j][k].s1/Umu[i+1][j][k].s0, \
                         0.5 );


//   real Ux = poly3_int( 0.0, 1.0, 2.0, Umu[i][j][k].s1/Umu[i][j][k].s0,  Umu[i+1][j][k].s1/Umu[i+1][j][k].s0, Umu[i+2][j][k].s1/Umu[i+2][j][k].s0, 0.5 );
//   real Jp = AR * Ux;
//   real Jm = AL * Ux;


   return 0.5 * ( Jp + Jm ) - 0.5*lam*( AR - AL );
}

real HY( local real AB [BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
         local real CS2[BSZ][BSZ][BSZ], real tau, int i, int j, int k, int I, int J, int K ){
   real DA0, DA1;
//   if( J==0 ){ 
//     AB[i][1][k] = poly3_int( 2.0, 3.0, 4.0, AB[j][2][k], AB[i][3][k], AB[i][4][k], 1.0 );
//     AB[i][0][k] = poly3_int( 1.0, 2.0, 3.0, AB[j][1][k], AB[i][2][k], AB[i][3][k], 0.0 );
//   }
//   if( J==NY-1 ){
//     AB[i][BSZ-2][k] = poly3_int( BSZ-3.0,BSZ-4.0, BSZ-5.0, AB[i][BSZ-3][k], AB[i][BSZ-4][k], AB[i][BSZ-5][k], BSZ-2.0 );
//     AB[i][BSZ-1][k] = poly3_int( BSZ-2.0,BSZ-3.0, BSZ-4.0, AB[i][BSZ-2][k], AB[i][BSZ-3][k], AB[i][BSZ-4][k], BSZ-1.0 );
//   }

   //// DAX(i, j, k)
   DA0 = minmod( theta*(AB[i][j+1][k]-AB[i][j][k]), \
         0.5*( AB[i][j+1][k] - AB[i][j-1][k] ), \
         theta*(AB[i][j][k]-AB[i][j-1][k]) );
   //if( J==0 ) DA0 = (AB[i][j+1][k]-AB[i][j][k]);

   DA1 = minmod( theta*(AB[i][j+2][k]-AB[i][j+1][k]), \
         0.5*( AB[i][j+2][k] - AB[i][j][k] ), \
         theta*(AB[i][j+1][k]-AB[i][j][k]) );
  
   //if( J==NY-1 ) DA1 = (AB[i][j][k]-AB[i][j-1][k]);

   real lam0 = maxPropagationSpeed( Umu[i][j][k].s0, Umu[i][j][k].s2, CS2[i][j][k] ) ;
   real lam1 = maxPropagationSpeed( Umu[i][j+1][k].s0, Umu[i][j+1][k].s2, CS2[i][j+1][k] ) ;
//   real lam = 0.5*( lam0 + lam1 );
  real lam = lam0;
//   real lam = max( lam0, lam1 );
   real  AL = AB[i][j][k]   + 0.5 * DA0;
   real  AR = AB[i][j+1][k] - 0.5 * DA1;

//   real Jp = lin_int( 0.0, 1.0, AB[i][j][k]*Umu[i][j][k].s2/Umu[i][j][k].s0, \
//                         AB[i][j+1][k]*Umu[i][j+1][k].s2/Umu[i][j+1][k].s0, \
//                         0.5 + DT/DY*lam );
//   real Jm = lin_int( 0.0, 1.0, AB[i][j][k]*Umu[i][j][k].s2/Umu[i][j][k].s0, \
//                         AB[i][j+1][k]*Umu[i][j+1][k].s2/Umu[i][j+1][k].s0, \
//                         0.5 - DT/DY*lam );

   real Jp = AR * lin_int( 0.0, 1.0, Umu[i][j][k].s2/Umu[i][j][k].s0, \
                         Umu[i][j+1][k].s2/Umu[i][j+1][k].s0, \
                         0.5 );
   real Jm = AL * lin_int( 0.0, 1.0, Umu[i][j][k].s2/Umu[i][j][k].s0, \
                         Umu[i][j+1][k].s2/Umu[i][j+1][k].s0, \
                         0.5 );
//   real Uy = poly3_int( 0.0, 1.0, 2.0, Umu[i][j][k].s2/Umu[i][j][k].s0,  Umu[i][j+1][k].s2/Umu[i][j+1][k].s0, Umu[i][j+2][k].s2/Umu[i][j+2][k].s0, 0.5 );
//   real Jp = AR * Uy;
//   real Jm = AL * Uy;

   return 0.5 * ( Jp + Jm ) - 0.5*lam*( AR - AL );
}

real HZ( local real AB [BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
         local real CS2[BSZ][BSZ][BSZ],  real tau, int i, int j, int k, int I, int J, int K ){
   real DA0, DA1;

   //// DAX(i, j, k)
   DA0 = minmod( theta*(AB[i][j][k+1]-AB[i][j][k]), \
         0.5*( AB[i][j][k+1] - AB[i][j][k-1] ), \
         theta*(AB[i][j][k]-AB[i][j][k-1]) );

   //// DAX(i+1, j, k)
   DA1 = minmod( theta*(AB[i][j][k+2]-AB[i][j][k+1]), \
         0.5*( AB[i][j][k+2] - AB[i][j][k] ), \
         theta*(AB[i][j][k+1]-AB[i][j][k]) );
  
   real lam = maxPropagationSpeed( Umu[i][j][k].s0, tau*Umu[i][j][k].s3, CS2[i][j][k] )/tau ;

   real  AL = AB[i][j][k]   + 0.5 * DA0;
   real  AR = AB[i][j][k+1] - 0.5 * DA1;

//   real Jp = lin_int( 0.0, 1.0, AB[i][j][k]*Umu[i][j][k].s3/Umu[i][j][k].s0, \
//                         AB[i][j][k+1]*Umu[i][j][k+1].s3/Umu[i][j][k+1].s0, \
//                         0.5 + DT/DZ*lam );
//   real Jm = lin_int( 0.0, 1.0, AB[i][j][k]*Umu[i][j][k].s3/Umu[i][j][k].s0, \
//                         AB[i][j][k+1]*Umu[i][j][k+1].s3/Umu[i][j][k+1].s0, \
//                         0.5 - DT/DZ*lam );

   real Jp = AR * lin_int( 0.0, 1.0, Umu[i][j][k].s3/Umu[i][j][k].s0, \
                         Umu[i][j][k+1].s3/Umu[i][j][k+1].s0, \
                         0.5 );
   real Jm = AL * lin_int( 0.0, 1.0, Umu[i][j][k].s3/Umu[i][j][k].s0, \
                         Umu[i][j][k+1].s3/Umu[i][j][k+1].s0, \
                         0.5 );


   return 0.5 * ( Jp + Jm ) - 0.5*lam*( AR - AL );
}



inline real KT3D( local real AB[BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
                    local real  CS2[BSZ][BSZ][BSZ], real tau, int i, int j, int k,
                    int I, int J, int K)
{
     /** \note the boundary condition is different here check DA1*/
     //real cs2 = P( Ed[i][j][k] )/max( Ed[i][j][k], acu );
     return AB[i][j][k] - DT/DX*( HX(AB, Umu, CS2, tau, i, j, k, I, J, K) - HX(AB, Umu, CS2, tau, i-1, j, k, I, J, K) ) \
                        - DT/DY*( HY(AB, Umu, CS2, tau, i, j, k, I, J, K) - HY(AB, Umu, CS2, tau, i, j-1, k, I, J, K) ) \
                        - DT/DZ*( HZ(AB, Umu, CS2, tau, i, j, k, I, J, K) - HZ(AB, Umu, CS2, tau, i, j, k-1, I, J, K) ) ;
}

inline real Upwind1( local real AB[BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
                    local real  CS2[BSZ][BSZ][BSZ], real tau, int i, int j, int k,
                    int I, int J, int K)
{
     real4 V = Umu[i][j][k] / Umu[i][j][k].s0 ;
     real  wx[3] = { max((real)(V.s1*DT/DX), (real)0.0),   1.0-fabs(V.s1*DT/DX),   min((real)(V.s1*DT/DX), (real) 0.0) };
     real  wy[3] = { max((real)(V.s2*DT/DY), (real)0.0),   1.0-fabs(V.s2*DT/DY),   min((real)(V.s2*DT/DY), (real) 0.0) };
     real  wz[3] = { max((real)(V.s3*DT/DZ), (real)0.0),   1.0-fabs(V.s3*DT/DZ),   min((real)(V.s3*DT/DZ), (real) 0.0) };
     
     return wx[0]*wy[0]*wz[0]*AB[i-1][j-1][k-1] \
         +  wx[0]*wy[0]*wz[1]*AB[i-1][j-1][k  ] \
         +  wx[0]*wy[0]*wz[2]*AB[i-1][j-1][k+1] \
         +  wx[0]*wy[1]*wz[0]*AB[i-1][j  ][k-1] \
         +  wx[0]*wy[1]*wz[1]*AB[i-1][j  ][k  ] \
         +  wx[0]*wy[1]*wz[2]*AB[i-1][j  ][k+1] \
         +  wx[0]*wy[2]*wz[0]*AB[i-1][j+1][k-1] \
         +  wx[0]*wy[2]*wz[1]*AB[i-1][j+1][k  ] \
         +  wx[0]*wy[2]*wz[2]*AB[i-1][j+1][k+1] \
         +  wx[1]*wy[0]*wz[0]*AB[i  ][j-1][k-1] \
         +  wx[1]*wy[0]*wz[1]*AB[i  ][j-1][k  ] \
         +  wx[1]*wy[0]*wz[2]*AB[i  ][j-1][k+1] \
         +  wx[1]*wy[1]*wz[0]*AB[i  ][j  ][k-1] \
         +  wx[1]*wy[1]*wz[1]*AB[i  ][j  ][k  ] \
         +  wx[1]*wy[1]*wz[2]*AB[i  ][j  ][k+1] \
         +  wx[1]*wy[2]*wz[0]*AB[i  ][j+1][k-1] \
         +  wx[1]*wy[2]*wz[1]*AB[i  ][j+1][k  ] \
         +  wx[1]*wy[2]*wz[2]*AB[i  ][j+1][k+1] \
         +  wx[2]*wy[0]*wz[0]*AB[i+1][j-1][k-1] \
         +  wx[2]*wy[0]*wz[1]*AB[i+1][j-1][k  ] \
         +  wx[2]*wy[0]*wz[2]*AB[i+1][j-1][k+1] \
         +  wx[2]*wy[1]*wz[0]*AB[i+1][j  ][k-1] \
         +  wx[2]*wy[1]*wz[1]*AB[i+1][j  ][k  ] \
         +  wx[2]*wy[1]*wz[2]*AB[i+1][j  ][k+1] \
         +  wx[2]*wy[2]*wz[0]*AB[i+1][j+1][k-1] \
         +  wx[2]*wy[2]*wz[1]*AB[i+1][j+1][k  ] \
         +  wx[2]*wy[2]*wz[2]*AB[i+1][j+1][k+1] ;
}

inline real Upwind( local real AB[BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
                    local real  CS2[BSZ][BSZ][BSZ], real tau, int i, int j, int k,
                    int I, int J, int K)
{
     real4 V = Umu[i][j][k] / Umu[i][j][k].s0 ;
     real  ax[2] = { min((real)(V.s1*DT/DX), (real) 0.0),   max((real)(V.s1*DT/DX),(real) 0.0) };
     real  ay[2] = { min((real)(V.s2*DT/DY), (real) 0.0),   max((real)(V.s2*DT/DY),(real) 0.0) };
     real  az[2] = { min((real)(V.s3*DT/DZ), (real) 0.0),   max((real)(V.s3*DT/DZ),(real) 0.0) };

     
//     real Ax[2] = {AB[i+1][j  ][k  ] - AB[i  ][j  ][k  ],  AB[i  ][j  ][k  ] - AB[i-1][j  ][k  ]} ;
//     real Ay[2] = {AB[i  ][j+1][k  ] - AB[i  ][j  ][k  ],  AB[i  ][j  ][k  ] - AB[i  ][j-1][k  ]} ;
//     real Az[2] = {AB[i  ][j  ][k+1] - AB[i  ][j  ][k  ],  AB[i  ][j  ][k  ] - AB[i  ][j  ][k-1]} ;
     
     real Ax[2] = {0.5*(4.0*AB[i+1][j  ][k  ] - 3.0 * AB[i  ][j  ][k  ] - AB[i+2][j  ][k  ] ),  0.5*(3.0*AB[i  ][j  ][k  ] - 4.0*AB[i-1][j  ][k  ] + AB[i-2][j  ][k  ])} ;
     real Ay[2] = {0.5*(4.0*AB[i  ][j+1][k  ] - 3.0 * AB[i  ][j  ][k  ] - AB[i  ][j+2][k  ] ),  0.5*(3.0*AB[i  ][j  ][k  ] - 4.0*AB[i  ][j-1][k  ] + AB[i  ][j-2][k  ])} ;
     real Az[2] = {0.5*(4.0*AB[i  ][j  ][k+1] - 3.0 * AB[i  ][j  ][k  ] - AB[i  ][j  ][k+2] ),  0.5*(3.0*AB[i  ][j  ][k  ] - 4.0*AB[i  ][j  ][k-1] + AB[i  ][j  ][k-2])} ;
     return AB[i][j][k] - ( ax[0] * Ax[0] + ax[1]*Ax[1] ) \
                        - ( ay[0] * Ay[0] + ay[1]*Ay[1] ) \
                        - ( az[0] * Az[0] + az[1]*Az[1] ) ;

}



#endif
