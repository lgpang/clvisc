#include<helper.h>
//#include<MWC64X.cl>
//constant int INDEX[4][4] = \
//{   0, 1, 2, 3, 
//    1, 4, 5, 6,
//    2, 5, 7, 8,
//    3, 6, 8, 9};  

//////////////////////////////////////////////////////////
/*! */

/** Read global data 12 A = pi^{mu nu} - v^{nu} * pi^{tau nu } to local memory 
 *  To calculate the source term from pi^{munu } for T^{mu nu}*/
void loadDiff( 
    __global real4 * d_Umu1,
    __global real  * d_pi1,
    __local real  A[12][VSZ][VSZ][VSZ],
    __private real pimn[10],
    int i, int j, int k, int IND)
{
  real4 Umu  =  d_Umu1[IND];
  real4 Vmu  =  Umu / Umu.s0;
  for( int m=0; m<10; m++ ){
    pimn[m] = d_pi1[ 10*IND + m ];
  }

  A[0][i][j][k] = pimn[ idx(0, 1) ] - Vmu.s1 * pimn[ idx(0, 0) ] ;
  A[1][i][j][k] = pimn[ idx(0, 2) ] - Vmu.s2 * pimn[ idx(0, 0) ] ;
  A[2][i][j][k] = pimn[ idx(0, 3) ] - Vmu.s3 * pimn[ idx(0, 0) ] ;

  A[3][i][j][k] = pimn[ idx(1, 1) ] - Vmu.s1 * pimn[ idx(0, 1) ] ;
  A[4][i][j][k] = pimn[ idx(1, 2) ] - Vmu.s2 * pimn[ idx(0, 1) ] ;
  A[5][i][j][k] = pimn[ idx(1, 3) ] - Vmu.s3 * pimn[ idx(0, 1) ] ;

  A[6][i][j][k] = pimn[ idx(2, 1) ] - Vmu.s1 * pimn[ idx(0, 2) ] ;
  A[7][i][j][k] = pimn[ idx(2, 2) ] - Vmu.s2 * pimn[ idx(0, 2) ] ;
  A[8][i][j][k] = pimn[ idx(2, 3) ] - Vmu.s3 * pimn[ idx(0, 2) ] ;

  A[9][i][j][k] = pimn[ idx(3, 1) ] - Vmu.s1 * pimn[ idx(0, 3) ] ;
  A[10][i][j][k]= pimn[ idx(3, 2) ] - Vmu.s2 * pimn[ idx(0, 3) ] ;
  A[11][i][j][k]= pimn[ idx(3, 3) ] - Vmu.s3 * pimn[ idx(0, 3) ] ;

}

inline real PartialX(__local real A1[12][VSZ][VSZ][VSZ], int n, int i, int j, int k, int I)
{
//  if( I== 0 ){
//    A1[n][0][j][k] = 2.0* A1[n][1][j][k] - A1[n][2][j][k];
//  }
//  else if( I== NX-1 ){
//    A1[n][VSZ-1][j][k] = 2.0*A1[n][VSZ-2][j][k] - A1[n][VSZ-3][j][k];
//  }

    return minmod( theta*(A1[n][i][j][k]-A1[n][i-1][j][k]), \
        0.5*(A1[n][i+1][j][k]-A1[n][i-1][j][k]), \
        theta*(A1[n][i+1][j][k]-A1[n][i][j][k]))/DX;
}

inline real PartialY(__local real A1[12][VSZ][VSZ][VSZ], int n, int i, int j, int k, int J)
{
//  if( J== 0 ){
//    A1[n][i][0][k] = 2.0* A1[n][i][1][k] - A1[n][i][2][k];
//  }
//  else if( J== NY-1 ){ 
//    A1[n][i][VSZ-1][k] = 2.0*A1[n][i][VSZ-2][k] - A1[n][i][VSZ-3][k];
//  }


    return minmod( theta*(A1[n][i][j][k]-A1[n][i][j-1][k]), \
        0.5*(A1[n][i][j+1][k]-A1[n][i][j-1][k]), \
        theta*(A1[n][i][j+1][k]-A1[n][i][j][k]))/DY;
}

inline real PartialZ(__local real A1[12][VSZ][VSZ][VSZ], int n, int i, int j, int k, int K)
{
  return minmod( theta*(A1[n][i][j][k]-A1[n][i][j][k-1]), \
      0.5*(A1[n][i][j][k+1]-A1[n][i][j][k-1]), \
      theta*(A1[n][i][j][k+1]-A1[n][i][j][k]))/DZ;
}


__kernel void updateSrcFromPimn(  
    __global real  * d_pi1,
    __global real4 * d_Umu1,            
    __global real4 * d_Src,
    const real tau,
    const int  Size)
{
  int I = get_global_id(0);
  int J = get_global_id(1);
  int K = get_global_id(2);

  //load Ed Umu to shared momory
  int i = get_local_id(0) + 1;
  int j = get_local_id(1) + 1;
  int k = get_local_id(2) + 1;

  /** halo cells i=0, VSZ-1 
   *  We need 2 halo cells to calculate ( pi^{mu nu} - v^{nu} * pi^{tau nu} ),nu
   *  */
  __local real  A[12][VSZ][VSZ][VSZ];

  int IND ; 

  real  pimn[10];
  IND = I*NY*NZ + J*NZ + K;
  loadDiff( d_Umu1, d_pi1, A, pimn, i, j, k, IND ); 

  // halo cells for i==0 and i== VSZ-1 and VSZ-2
  if( get_local_id(0) == 0 ){
    /// left: for 2 left most cells use constant extrapolation
    IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-1)*NY*NZ + J*NZ + K );
    loadDiff( d_Umu1, d_pi1, A, pimn, 0, j, k, IND ); 
    /// right: for 2 right most cells use constant extrapolation
    IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+VSZ-2)*NY*NZ + J*NZ + K );
    loadDiff( d_Umu1, d_pi1, A,  pimn,VSZ-1, j, k, IND ); 
  }

  // halo cells for j==0 and j==VSZ-1
  if( get_local_id(1) == 0 ){
    IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-1)*NZ + K);
    loadDiff( d_Umu1, d_pi1, A, pimn, i, 0, k, IND ); 
    IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+VSZ-2)*NZ + K);
    loadDiff( d_Umu1, d_pi1, A, pimn, i, VSZ-1, k, IND ); 
  }
  // halo cells for k==0 and k==VSZ-1
  if( get_local_id(2) == 0 ){
    IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-1);
    loadDiff( d_Umu1, d_pi1, A, pimn, i, j, 0, IND ); 
    IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+VSZ-2);
    loadDiff( d_Umu1, d_pi1, A, pimn, i, j, VSZ-1, IND ); 
  }

  barrier( CLK_LOCAL_MEM_FENCE );

  real4 src;

  src.s0 =  tau * ( PartialX( A, 0, i, j, k, I) + PartialY( A, 1, i, j, k, J) + PartialZ( A, 2, i, j, k, K) )  + pimn[0] + tau*tau*pimn[9] ;
  src.s1 =  tau * ( PartialX( A, 3, i, j, k, I) + PartialY( A, 4, i, j, k, J) + PartialZ( A, 5, i, j, k, K) )  + pimn[1];
  src.s2 =  tau * ( PartialX( A, 6, i, j, k, I) + PartialY( A, 7, i, j, k, J) + PartialZ( A, 8, i, j, k, K) )  + pimn[2];
  src.s3 =  tau * ( PartialX( A, 9, i, j, k, I) + PartialY( A,10, i, j, k, J) + PartialZ( A, 11,i, j, k, K) )  + 3.0 * pimn[3];

  d_Src[ IND ] = d_Src[IND] + src;

}
