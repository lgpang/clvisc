#include<helper.h>

//////////////////////////////////////////////////////////
/*! */
inline real gammav( real4 * v , real * tau );

inline real KT1D(local real AB[BSZ][BSZ][BSZ],  local real4 Umu[BSZ][BSZ][BSZ], \
                    local real  CS2[BSZ][BSZ][BSZ], real tau, int i, int j, int k, int I, int J, int K );

/** solve energy density from T00 and K=sqrt(T01**2 + T02**2 + T03**2)
 * */

__kernel void kt_src_alongx(
                     __global real4 * d_Src,     // out put
		     __global real4 * d_ev,
		     const real time,
		     const int step) {
    // store one line of data in local memory
    __local real4 ev[NX+4];
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    int i = get_local_id(0) + 2;

    int IND = I*NY*NZ + J*NZ + K;
    // load 1D data to local memory
    ev[i] = d_ev[IND];
    if ( i == 2 ) {
       ev[0] = d_ev[J*NZ+K];
       ev[1] = ev[0];
       ev[NX+3] = d_ev[NX*NY*NZ+J*NZ+K];
       ev[NX+2] = ev[NX+3];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    if ( step == 1 ) d_Src[IND] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    // update d_Src according to kt_1d_alongx
}

__kernel void stepUpdate(
	__global real4 * d_Tm00,            
	__global real4 * d_Tm01,            
	__global real4 * d_Umu1,  /**< Umu at time   */          
    __global real4 * d_Src,   /**< Source term from viscous contribution */
	__global real  * d_Ed ,           
    __global real4 * d_NewTm00,
    __global real4 * d_NewUmu,
    __global real * d_NewEd,
    const real tau,
    const int  halfStep,
	const int  Size)
{
    real dt = DT;
    //if( halfStep ) dt = 0.5*DT;
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    //load Ed Umu to shared momory
    int i = get_local_id(0) + 2;
    int j = get_local_id(1) + 2;
    int k = get_local_id(2) + 2;

    /** halo cells i=0, i=1, BSZ-2, BSZ-1 
     *  We need 2 halo cells in the right for KT algorithm*/
    __local real T00[BSZ][BSZ][BSZ];
    __local real T01[BSZ][BSZ][BSZ];
    __local real T02[BSZ][BSZ][BSZ];
    __local real T03[BSZ][BSZ][BSZ];

    __local real4 Umu[BSZ][BSZ][BSZ];
    __local real   Ed[BSZ][BSZ][BSZ];
    __local real   Pr[BSZ][BSZ][BSZ];
    __local real  CS2[BSZ][BSZ][BSZ];

    //if( I>NX-1 && J>NY-1 && K>NZ-1 ) return ;

    int IND = I*NY*NZ + J*NZ + K;
    loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, i, j, k, IND ); 

     // halo cells for i==0 and i== BSZ-1 and BSZ-2
    if( get_local_id(0) == 0 ){
        /// left: for 2 left most cells use constant extrapolation
        IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-2)*NY*NZ + J*NZ + K );
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, 0, j, k, IND ); 

        IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-1)*NY*NZ + J*NZ + K );
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, 1, j, k, IND ); 

        /// right: for 2 right most cells use constant extrapolation
        IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-4)*NY*NZ + J*NZ + K );
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, BSZ-2, j, k, IND ); 
        IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-3)*NY*NZ + J*NZ + K );
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2,  BSZ-1, j, k, IND ); 

    }

    // halo cells for j==0 and j==BSZ-1
    if( get_local_id(1) == 0 ){
        IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-2)*NZ + K);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, i, 0, k, IND ); 
        IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-1)*NZ + K);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, i, 1, k, IND ); 

        IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-4)*NZ + K);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2,  i, BSZ-2, k, IND ); 
        IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-3)*NZ + K);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2,  i, BSZ-1, k, IND ); 

    }
    // halo cells for k==0 and k==BSZ-1
    if( get_local_id(2) == 0 ){
        IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-2);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, i, j, 0, IND ); 
        IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-1);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2, i, j, 1, IND ); 

        IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-4);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2,  i, j, BSZ-2, IND ); 
        IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-3);
        loadDataIdeal( d_Tm00, d_Umu1, d_Ed, T00, T01, T02, T03, Umu, Ed, Pr, CS2,  i, j, BSZ-1, IND ); 
    }
    barrier( CLK_LOCAL_MEM_FENCE );
 
    IND = I*NY*NZ + J*NZ + K;
    real4 Src = d_Src[ IND ];

    real time = tau + dt;   // time after evolve
    real4 F0 = d_Tm01[ IND ] ;


    real time1 = tau+(1-halfStep)*DT;

	CalcSources( &Src, Pr, Umu, F0.s0, F0.s3, time1, i, j, k, I, J, K);
	//CalcSources( &Src, Pr, Umu, F0.s0, F0.s3, tau-halfStep*DT, i, j, k, I, J, K);

    //Use KT algorithm to update
    real TH00 = (KT3D( T00, Umu, CS2, tau, i, j, k, I, J, K ) - dt*Src.s0)/time ;
    real TH01 = (KT3D( T01, Umu, CS2, tau, i, j, k, I, J, K ) - dt*Src.s1)/time ;
    real TH02 = (KT3D( T02, Umu, CS2, tau, i, j, k, I, J, K ) - dt*Src.s2)/time ;
    real TH03 = (KT3D( T03, Umu, CS2, tau, i, j, k, I, J, K ) - dt*Src.s3)      ;


    TH00 = max( acu, TH00 );

    //TH01 = (fabs(TH01)>acu) ? TH01 : 0.0;
    if( fabs(TH01)< acu ) TH01 = 0.0;
    if( fabs(TH02)< acu ) TH02 = 0.0;
    if( fabs(TH03)< acu ) TH03 = 0.0;

    /*  \note Set TH03 here for test */
    //TH03 = 0.0;

    real M = sqrt( TH01*TH01 + TH02*TH02 + TH03*TH03 );

    if( M>TH00 ) TH00 = 1.000001 * M;

    //rootFinding to get new Ed, Umu
    real EdFind;
    //real K2 = M*M;
    real K2 = TH01*TH01 + TH02*TH02 + TH03*TH03 ;
    rootFinding( &EdFind, &TH00, &K2 );
    EdFind = max( acu, EdFind );

    real PR = P( EdFind );

    //Why this is much faster than max(acu, M) ?
    real EPV = max(acu, TH00 + PR);

    real4 vi = (real4) ( 1.0, TH01/EPV, TH02/EPV, TH03/EPV/time );

    real4 umu = gammav( & vi, & time ) * vi;

    //if( K > NX/2  && umu.s3 < 0 ) umu.s3 = fabs( umu.s3 );

    real epp = EdFind + PR;

    d_NewEd[IND] = EdFind;
    d_NewUmu[IND] = umu;
    d_NewTm00[IND] = time*( epp*umu*umu.s0  - PR*(real4)(1.0, 0.0, 0.0, 0.0) );

}

__kernel void updateGlobalMem(
    __global real4 * d_Tm00, 
    __global real4 * d_Tm01, 
    __global real4 * d_Umu1, 
    __global real * d_Ed, 
    __global real4 * d_NewTm00, 
    __global real4 * d_NewUmu, 
    __global real * d_NewEd,
    const  real   tau,
    const    int  halfStep,
    const    int  Size )
{

    int IND = get_global_id(0);

    real time = tau + DT;

    real4 Tm01 = 0.5*(d_Tm00[IND] + d_Tm01[IND]);
    //Use KT algorithm to update
    real TH00 = Tm01.s0/time ;
    real TH01 = Tm01.s1/time ;
    real TH02 = Tm01.s2/time ;
    real TH03 = Tm01.s3      ;

    TH00 = max( acu, TH00 );
    //TH01 = (fabs(TH01)>acu) ? TH01 : 0.0;

    if( fabs(TH01)< acu ) TH01 = 0.0;
    if( fabs(TH02)< acu ) TH02 = 0.0;
    if( fabs(TH03)< acu ) TH03 = 0.0;

    /* \note set TH03 = 0.0 for test */
    //TH03 = 0.0;

    real M = sqrt( TH01*TH01 + TH02*TH02 + TH03*TH03 );

    if( M>TH00 ) TH00 = 1.000001 * M;

    //rootFinding to get new Ed, Umu
    real EdFind;
    real K2 = M*M;
    rootFinding( &EdFind, &TH00, &K2 );
    EdFind = max( acu, EdFind );

    real PR = P( EdFind );

    //Why this is much faster than max(acu, M) ?
    real EPV = max( acu, TH00 + PR );

    real4 vi = (real4) ( 1.0, TH01/EPV, TH02/EPV, TH03/EPV/time );

    real4 umu = gammav( & vi, & time ) * vi;

    umu.s0 = sqrt( 1.0 + umu.s1*umu.s1 + umu.s2*umu.s2 + time*time*umu.s3*umu.s3 );

    real epp = EdFind + PR ;

    d_Ed[IND] = EdFind;
    d_Umu1[IND] = umu;
    d_Tm01[IND] = time*( epp*umu*umu.s0  - PR*(real4)(1.0, 0.0, 0.0, 0.0) );
    d_Tm00[IND] = time*( epp*umu*umu.s0  - PR*(real4)(1.0, 0.0, 0.0, 0.0) );

    /** Set up boundary condition here */
}


inline real gammav( real4 * v , real * tau ){
    //return 1.0/sqrt(max(1.0E-4, 1.0 - (*v).s1*(*v).s1 - (*v).s2*(*v).s2 - (*tau)*(*tau)*(*v).s3*(*v).s3 ));
    return (real)1.0/sqrt(max(acu*acu, (real)1.0 - (*v).s1*(*v).s1 - (*v).s2*(*v).s2 - (*tau)*(*tau)*(*v).s3*(*v).s3 ));
}


inline void CalcSources( real4 * Src, __local real Pr[BSZ][BSZ][BSZ], __local real4 Umu[BSZ][BSZ][BSZ] , \
     real F00, real F03, real time, int i, int j, int k, int I, int J, int K )
{
/** \bug Notice the extropolation will make the program crash soon */
//   if( I==0 ){ 
//     Pr[1][j][k] = poly3_int( 2.0, 3.0, 4.0, Pr[2][j][k], Pr[3][j][k], Pr[4][j][k], 1.0 );
//     Pr[0][j][k] = poly3_int( 1.0, 2.0, 3.0, Pr[1][j][k], Pr[2][j][k], Pr[3][j][k], 0.0 );
//   }
//   if( I==NX-1 ){
//     Pr[BSZ-2][j][k] = poly3_int( BSZ-3.0,BSZ-4.0, BSZ-5.0, Pr[BSZ-3][j][k], Pr[BSZ-4][j][k], Pr[BSZ-5][j][k], BSZ-2.0 );
//     Pr[BSZ-1][j][k] = poly3_int( BSZ-2.0,BSZ-3.0, BSZ-4.0, Pr[BSZ-2][j][k], Pr[BSZ-3][j][k], Pr[BSZ-4][j][k], BSZ-1.0 );
//   }
//   if( J==0 ){ 
//     Pr[i][1][k] = poly3_int( 2.0, 3.0, 4.0, Pr[j][2][k], Pr[i][3][k], Pr[i][4][k], 1.0 );
//     Pr[i][0][k] = poly3_int( 1.0, 2.0, 3.0, Pr[j][1][k], Pr[i][2][k], Pr[i][3][k], 0.0 );
//   }
//   if( J==NY-1 ){
//     Pr[i][BSZ-2][k] = poly3_int( BSZ-3.0,BSZ-4.0, BSZ-5.0, Pr[i][BSZ-3][k], Pr[i][BSZ-4][k], Pr[i][BSZ-5][k], BSZ-2.0 );
//     Pr[i][BSZ-1][k] = poly3_int( BSZ-2.0,BSZ-3.0, BSZ-4.0, Pr[i][BSZ-2][k], Pr[i][BSZ-3][k], Pr[i][BSZ-4][k], BSZ-1.0 );
//   }
//
    real px, py, pz, pvx, pvy, pvz;
    /** pressure at i,j,k will be used many times */
    real pr   = Pr[i][j][k] ;
    real pip1 = Pr[i+1][j][k];
    real pjp1 = Pr[i][j+1][k];
    real pkp1 = Pr[i][j][k+1];

    real pim1 = Pr[i-1][j][k];
    real pjm1 = Pr[i][j-1][k];
    real pkm1 = Pr[i][j][k-1];

    px = minmod( theta*( pip1-pr ), 0.5*( pip1 - pim1 ), theta*( pr-pim1 ) ) / DX;
    py = minmod( theta*( pjp1-pr ), 0.5*( pjp1 - pjm1 ), theta*( pr-pjm1 ) ) / DY;
    pz = minmod( theta*( pkp1-pr ), 0.5*( pkp1 - pkm1 ), theta*( pr-pkm1 ) ) / DZ;

    //real time0 = time - 0.5*DT;  // source term at (time step n and n+1/2 )
    real time0 = time ;  // source term at (time step n and n+1 )
    /** Release px, py, pz and F03 earlier in each threads */
    (*Src).s1 += time0 * px;
    (*Src).s2 += time0 * py;
    (*Src).s3 += ( pz + 2.0*F03 ) / time0;

    real vx   = Umu[i][j][k].s1/Umu[i][j][k].s0 ; 
    real vxm1 = Umu[i-1][j][k].s1/Umu[i-1][j][k].s0 ;
    real vxp1 = Umu[i+1][j][k].s1/Umu[i+1][j][k].s0 ;

    pvx = minmod( theta*( pip1 * vxp1 - pr*vx ), 0.5*( pip1*vxp1 - pim1*vxm1 ),  theta*( pr*vx - pim1*vxm1 ) ) / DX;

    real vy   = Umu[i][j][k].s2/Umu[i][j][k].s0 ; 
    real vym1 = Umu[i][j-1][k].s2/Umu[i][j-1][k].s0 ;
    real vyp1 = Umu[i][j+1][k].s2/Umu[i][j+1][k].s0 ;
    pvy = minmod( theta*( pjp1 * vyp1 - pr*vy ), 0.5*( pjp1 * vyp1 - pjm1*vym1 ),  theta*( pr*vy - pjm1*vym1 ) ) /DY ;


    real vz   = Umu[i][j][k].s3/Umu[i][j][k].s0 ; 
    real vzm1 = Umu[i][j][k-1].s3/Umu[i][j][k-1].s0 ;
    real vzp1 = Umu[i][j][k+1].s3/Umu[i][j][k+1].s0 ;
    pvz = minmod( theta*( pkp1 * vzp1 - pr*vz ), 0.5*( pkp1 * vzp1 - pkm1 * vzm1 ),  theta*( pr*vz - pkm1*vzm1 ) ) /DZ;


    (*Src).s0 += pr + pown( vz, 2 ) * time0 * ( F00 + pr*time0 ) + time0*(pvx+pvy+pvz) ;

}


