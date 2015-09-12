#include<helper.h>

/** 1D linear interpolation for real4 type */
inline real4 lin_int4( real x1, real x2, real4 y1, real4 y2, real x )
{
    real r = ( x - x1 )/ (x2-x1 );
    return y1*(1.0-r) + y2*r ;
}

inline real4 bilinear_int(real x0,real x1,real y0, real y1, real4* f00, real4* f01, real4* f10, real4* f11, real x, real y)
{
  real4 temp=(real4)(0.0);
  real t=(x-x0)/(x1-x0);
  real u=(y-y0)/(y1-y0);

  if ((isfinite(u)==1)&&(isfinite(t)==1))
    {
      temp=(1-t)*(1-u)*(*f00)+t*(1-u)*(*f10)+t*u*(*f11)+(1-t)*u*(*f01);
    }
  else
    {
      if (isfinite(u)==0)
	temp=lin_int4(x0,x1,*f00,*f10,x);
      if (isfinite(t)==0)
	temp=lin_int4(y0,y1,*f00,*f01,y);
    }
  
  return temp;
}


real4 trilinear_int(real x0,real x1,real y0, real y1, real z0, real z1,\
                        real4* f000, real4* f001, real4* f010, real4* f011, \
                        real4* f100, real4* f101,real4* f110, real4* f111, \
                        real x, real y,real z)
{
  real4 temp= (real4)(0.0);
  real t=(x-x0)/(x1-x0);
  real u=(y-y0)/(y1-y0);
  real v=(z-z0)/(z1-z0);

  if ((isfinite(u)==1)&&(isfinite(t)==1)&&(isfinite(v)==1))
    {
      temp=(1-t)*(1-u)*(1-v)*(*f000);
      temp+=(1-t)*(1-u)*v*(*f001);
      temp+=(1-t)*u*(1-v)*(*f010);
      temp+=(1-t)*u*v*(*f011);
      temp+=t*(1-u)*(1-v)*(*f100);
      temp+=t*(1-u)*v*(*f101);
      temp+=t*u*(1-v)*(*f110);
      temp+=t*u*v*(*f111);
    }
  else
    {
      if (isfinite(t)==0)
	temp=bilinear_int(y0,y1,z0,z1,f000,f010,f011,f001,y,z);
      if (isfinite(v)==0)
	temp=bilinear_int(x0,x1,y0,y1,f000,f100,f110,f010,x,y);
      if (isfinite(u)==0)
	temp=bilinear_int(x0,x1,z0,z1,f000,f100,f101,f001,x,z);
    }
  
  return temp;
}



/** frac=0 for T<0.184 and frac=1 for T>0.220 */
__kernel void getBulkInfo(  
        __global real4  * bulkInfo,
        __global real   * d_Ed,
        __global real4  * d_Umu,
        const real tau,
        const int  Size)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int Ni = get_global_size(0);
    int Nj = get_global_size(1);

    real x = -9.75 + i * 0.3;
    real y = -9.75 + j * 0.3;
    real xmin = -NX/2*DX;
    real ymin = -NY/2*DY;
    int i0 = (int)( floor((x-(xmin))/DX )  ) ;
    int j0 = (int)( floor((y-(ymin))/DY )  ) ;
    int k0 = NZ/2;

    real Ed00 = d_Ed[ i0*NY*NZ + j0*NZ + k0   ];
    real4 U00 = d_Umu[ i0*NY*NZ + j0*NZ + k0     ];
    real Ed01 = d_Ed[ i0*NY*NZ + (j0+1)*NZ + k0 ];
    real4 U01 = d_Umu[ i0*NY*NZ + (j0+1)*NZ + k0  ];
    real Ed10 = d_Ed[ (i0+1)*NY*NZ + j0*NZ + k0 ];
    real4 U10 = d_Umu[ (i0+1)*NY*NZ + j0*NZ + k0 ];
    real Ed11 = d_Ed[ (i0+1)*NY*NZ + (j0+1)*NZ + k0  ];
    real4 U11 = d_Umu[ (i0+1)*NY*NZ + (j0+1)*NZ + k0  ];

    real4 f00 = (real4) (Ed00, T(Ed00), U00.s1/U00.s0, U00.s2/U00.s0 );
    real4 f01 = (real4) (Ed01, T(Ed01), U01.s1/U01.s0, U01.s2/U01.s0 );
    real4 f10 = (real4) (Ed10, T(Ed10), U10.s1/U10.s0, U10.s2/U10.s0 );
    real4 f11 = (real4) (Ed11, T(Ed11), U11.s1/U11.s0, U11.s2/U11.s0 );
    
    real  x0, x1, y0, y1;
    x0 = xmin +i0*DX; 
    y0 = ymin +j0*DY;
    x1 = xmin +(i0+1)*DX;
    y1 = ymin +(j0+1)*DY;
    bulkInfo[ i*Nj + j] = bilinear_int( x0, x1, y0, y1, &f00, &f01, &f10, &f11, x, y );
}


/** frac=0 for T<0.184 and frac=1 for T>0.220 */
__kernel void getBulkInfo3d(  
        __global real4  * bulkInfo,
        __global real   * d_Ed,
        __global real4  * d_Umu,
        const real tau,
        const int  Size)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int Ni = get_global_size(0);
    int Nj = get_global_size(1);
    int Nk = get_global_size(2);

    real x = i * 0.4;
    real y = j * 0.4;
    real z = k * 0.4;
    real xmin = -NX/2*DX;
    real ymin = -NY/2*DY;
    real zmin = -NZ/2*DZ;
    int i0 = (int)( floor((x-(xmin))/DX )  ) ;
    int j0 = (int)( floor((y-(ymin))/DY )  ) ;
    int k0 = (int)( floor((z-(zmin))/DZ )  ) ;

    real Ed000 = d_Ed[ i0*NY*NZ + j0*NZ + k0   ];
    real4 U000 = d_Umu[ i0*NY*NZ + j0*NZ + k0     ];
    real Ed001 = d_Ed[ i0*NY*NZ + j0*NZ + k0+1 ];
    real4 U001 = d_Umu[ i0*NY*NZ + j0*NZ + k0+1  ];

    real Ed010 = d_Ed[ i0*NY*NZ + (j0+1)*NZ + k0   ];
    real4 U010 = d_Umu[ i0*NY*NZ + (j0+1)*NZ + k0     ];
    real Ed011 = d_Ed[ i0*NY*NZ + (j0+1)*NZ + k0+1 ];
    real4 U011 = d_Umu[ i0*NY*NZ + (j0+1)*NZ + k0+1  ];

    real Ed100 = d_Ed[ (i0+1)*NY*NZ + j0*NZ + k0 ];
    real4 U100 = d_Umu[ (i0+1)*NY*NZ + j0*NZ + k0 ];
    real Ed101 = d_Ed[ (i0+1)*NY*NZ + j0*NZ + k0+1  ];
    real4 U101 = d_Umu[ (i0+1)*NY*NZ + j0*NZ + k0+1  ];

    real Ed110 = d_Ed[ (i0+1)*NY*NZ + (j0+1)*NZ + k0 ];
    real4 U110 = d_Umu[ (i0+1)*NY*NZ + (j0+1)*NZ + k0 ];
    real Ed111 = d_Ed[ (i0+1)*NY*NZ + (j0+1)*NZ + k0+1  ];
    real4 U111 = d_Umu[ (i0+1)*NY*NZ + (j0+1)*NZ + k0+1  ];

    //// Y = atanh(tanh(Y-etas))+etas; vz = tanhY ; 
    real Y = tanh(atanh(U000.s3/U000.s0*tau)+z    );
    real4 f000 = (real4) (T(Ed000), U000.s1/U000.s0/cosh(Y), U000.s2/U000.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U001.s3/U001.s0*tau)+z+0.4);
    real4 f001 = (real4) (T(Ed001), U001.s1/U001.s0/cosh(Y), U001.s2/U001.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U010.s3/U010.s0*tau)+z    );
    real4 f010 = (real4) (T(Ed010), U010.s1/U010.s0/cosh(Y), U010.s2/U010.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U011.s3/U011.s0*tau)+z+0.4);
    real4 f011 = (real4) (T(Ed011), U011.s1/U011.s0/cosh(Y), U011.s2/U011.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U100.s3/U100.s0*tau)+z    );
    real4 f100 = (real4) (T(Ed100), U100.s1/U100.s0/cosh(Y), U100.s2/U100.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U101.s3/U101.s0*tau)+z+0.4);
    real4 f101 = (real4) (T(Ed101), U101.s1/U101.s0/cosh(Y), U101.s2/U101.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U110.s3/U110.s0*tau)+z    );
    real4 f110 = (real4) (T(Ed110), U110.s1/U110.s0/cosh(Y), U110.s2/U110.s0/cosh(Y), tanh(Y) );
    Y = (atanh(U111.s3/U111.s0*tau)+z+0.4);
    real4 f111 = (real4) (T(Ed111), U111.s1/U111.s0/cosh(Y), U111.s2/U111.s0/cosh(Y), tanh(Y) );

    real  x0, x1, y0, y1, z0, z1;
    x0 = xmin +i0*DX; 
    y0 = ymin +j0*DY;
    z0 = zmin +k0*DZ;
    x1 = xmin +(i0+1)*DX;
    y1 = ymin +(j0+1)*DY;
    z1 = zmin +(k0+1)*DZ;
    
    bulkInfo[ i*Nj*Nk + j*Nk + k ] = trilinear_int( x0, x1, y0, y1, z0, z1, &f000, &f001,
                        &f010, &f011, &f100, &f101, &f110, &f111, x, y, z );
}
