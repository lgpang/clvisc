//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
inline float eps(float x, float y){
    return  pow(1 + (1.0/4.0)*pow(-pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2)/(pow(L0, 2)*pow(tau, 2)), -1.33333333333333 + 1.0/lam1)/pow(tau, 4) ;
}


inline float ut(float x, float y){
    return  pow(-4*pow(tau, 2)*(pow(x, 2) + pow(y, 2))/pow(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2), 2) + 1, -1.0/2.0) ;
}


inline float ux(float x, float y){
    return  2*tau*x/(sqrt(-4*pow(tau, 2)*(pow(x, 2) + pow(y, 2))/pow(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2), 2) + 1)*(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2))) ;
}


inline float uy(float x, float y){
    return  2*tau*y/(sqrt(-4*pow(tau, 2)*(pow(x, 2) + pow(y, 2))/pow(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2), 2) + 1)*(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2))) ;
}


inline float pitt(float x, float y){
    return  -4*pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)*(pow(x, 2) + pow(y, 2))*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(lam1*pow(tau, 2)*pow(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2), 2)) ;
}


inline float pitx(float x, float y){
    return  -2*x*pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))*(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2))/(lam1*pow(tau, 3)*pow(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2), 2)) ;
}


inline float pity(float x, float y){
    return  -2*y*pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))*(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2))/(lam1*pow(tau, 3)*pow(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2), 2)) ;
}


inline float pixx(float x, float y){
    // avoid the  NAN problem when x=0, y=0
    if ( fabs(x) < 1.0E-6 ) x = 1.0E-6;
    if ( fabs(y) < 1.0E-6 ) y = 1.0E-6;
    return  -pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))*(pow(x, 2)*pow(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2), 2) + pow(y, 2)*(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2)))/(lam1*pow(tau, 4)*(pow(x, 2) + pow(y, 2))*pow(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2), 2)) ;
}


inline float pixy(float x, float y){
    // avoid the  NAN problem when x=0, y=0
    if ( fabs(x) < 1.0E-6 ) x = 1.0E-6;
    if ( fabs(y) < 1.0E-6 ) y = 1.0E-6;
    return  x*y*pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))*(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2) - pow(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(lam1*pow(tau, 4)*(pow(x, 2) + pow(y, 2))*pow(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2), 2)) ;
}


inline float piyy(float x, float y){
    // avoid the  NAN problem when x=0, y=0
    if ( fabs(x) < 1.0E-6 ) x = 1.0E-6;
    if ( fabs(y) < 1.0E-6 ) y = 1.0E-6;
    return  -pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))*(pow(x, 2)*(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2)) + pow(y, 2)*pow(pow(L0, 2) + pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(lam1*pow(tau, 4)*(pow(x, 2) + pow(y, 2))*pow(4*pow(L0, 2)*(pow(x, 2) + pow(y, 2)) + pow(pow(L0, 2) + pow(tau, 2) - pow(x, 2) - pow(y, 2), 2), 2)) ;
}

/* pizz = pi^{\eta\eta} */
inline float pizz(float x, float y){
    return  2*pow((0.25)*(4*pow(L0, 2)*pow(tau, 2) + pow(pow(L0, 2) - pow(tau, 2) + pow(x, 2) + pow(y, 2), 2))/(pow(L0, 2)*pow(tau, 2)), (-1.33333333333333*lam1 + 1)/lam1)/(lam1*pow(tau, 6)) ;
}


__kernel void CreateIniCond( __global float4 * d_ev,
                             __global float * d_pi, 
                             const float pixx_, 
                             const float pixy_,
                             const float piyy_)
{
   int i = get_global_id(0);
   int j = get_global_id(1);
   int k = get_global_id(2);

   float x = ( i - NX/2 ) * DX;
   float y = ( j - NY/2 ) * DY;
   float z = ( k - NZ/2 ) * DZ;

   int IND = (i*NY*NZ + j*NZ + k);

   float u0 = ut(x, y);
   d_ev[IND] = (float4)(eps(x,y), ux(x, y)/u0, uy(x, y)/u0, 0.0f);

   d_pi[ 10*IND+ 0] = pitt(x, y);
   d_pi[ 10*IND+ 1] = pitx(x, y);
   d_pi[ 10*IND+ 2] = pity(x, y);
   d_pi[ 10*IND+ 3] = 0.0f;
   d_pi[ 10*IND+ 4] = pixx(x, y);
   d_pi[ 10*IND+ 5] = pixy(x, y);
   d_pi[ 10*IND+ 6] = 0.0f;
   d_pi[ 10*IND+ 7] = piyy(x, y);
   d_pi[ 10*IND+ 8] = 0.0f;
   d_pi[ 10*IND+ 9] = pizz(x, y);

   //if ( i == NX/2 && j == NY/2 ) {
   //    d_pi[ 10*IND+ 4] = pixx_;
   //    d_pi[ 10*IND+ 5] = pixy_;
   //    d_pi[ 10*IND+ 7] = piyy_;
   //}
}
