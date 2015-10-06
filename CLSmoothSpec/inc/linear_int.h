#ifndef  __LINEARINTP__
#define  __LINEARINTP__
#include <math.h>

double linear_int(double x0,double x1,double y0, double y1, double x)
{
  double temp=0;
  temp=((x-x0)*y1+(x1-x)*y0)/(x1-x0);
  return temp;
}


//inspired by numerical recipes
//x0,x1: grid points in x-direction
//y0,y1: grid points in y-direction
//f0-f3: function value starting at x0,y0, continue counterclockwise
//put differently: f0=f(x0,y0)
//f1=f(x1,y0)
//f2=f(x1,y1)
//f3=f(x0,y1)
double bilinear_int(double x0,double x1,double y0, double y1, double f0, double f1, double f2, double f3, double x, double y)
{
  double temp=0;
  double t=(x-x0)/(x1-x0);
  double u=(y-y0)/(y1-y0);

  if ((std::isfinite(u)==1)&&(std::isfinite(t)==1))
    {
      temp=(1-t)*(1-u)*f0+t*(1-u)*f1+t*u*f2+(1-t)*u*f3;
    }
  else
    {
      if (std::isfinite(u)==0)
	temp=linear_int(x0,x1,f0,f2,x);
      if (std::isfinite(t)==0)
	temp=linear_int(y0,y1,f0,f2,y);
    }
  
  

  //printf("t=%f u=%f f0=%f f1=%f f2=%f f3=%f temp=%f aha %i\n",t,u,f0,f1,f2,f3,temp,std::isfinite(t));
  return temp;
}


double trilinear_int(double x0,double x1,double y0, double y1, double z0, double z1,double f000, double f001, double f010, double f011, double f100, double f101,double f110, double f111, double x, double y,double z)
{
  double temp=0;
  double t=(x-x0)/(x1-x0);
  double u=(y-y0)/(y1-y0);
  double v=(z-z0)/(z1-z0);

  if ((std::isfinite(u)==1)&&(std::isfinite(t)==1)&&(std::isfinite(v)==1))
    {
      temp=(1-t)*(1-u)*(1-v)*f000;
      temp+=(1-t)*(1-u)*v*f001;
      temp+=(1-t)*u*(1-v)*f010;
      temp+=(1-t)*u*v*f011;
      temp+=t*(1-u)*(1-v)*f100;
      temp+=t*(1-u)*v*f101;
      temp+=t*u*(1-v)*f110;
      temp+=t*u*v*f111;
    }
  else
    {
      if (std::isfinite(t)==0)
	temp=bilinear_int(y0,y1,z0,z1,f000,f010,f011,f001,y,z);
      if (std::isfinite(v)==0)
	temp=bilinear_int(x0,x1,y0,y1,f000,f100,f110,f010,x,y);
      if (std::isfinite(u)==0)
	temp=bilinear_int(x0,x1,z0,z1,f000,f100,f101,f001,x,z);

    }
  
  

  //printf("t=%f u=%f f0=%f f1=%f f2=%f f3=%f temp=%f aha %i\n",t,u,f0,f1,f2,f3,temp,std::isfinite(t));
  return temp;
}
#endif
