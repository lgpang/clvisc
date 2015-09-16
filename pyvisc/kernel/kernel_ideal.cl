#include<helper.h>

#define ALONG_X 1
#define ALONG_Y 2
#define ALONG_Z 3
// output: d_Src; all the others are input
__kernel void kt_src_alongx(
                     __global real4 * d_Src,     
		     __global real4 * d_ev,
		     const real time,
		     const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    if ( I >= NX || J >= NY || K >= NZ ) return;

    int IND = I*NY*NZ + J*NZ + K;
    // store one line of data in local memory
    __local real4 ev[NX+4];
    // load 1D data to local memory
    for ( int i=get_local_id(0); i < NX; i += get_local_size(0) ) {
        ev[i+2] = d_ev[IND];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    // set boundary condition
    if ( (int)get_local_id(0) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NX+3] = ev[NX+1];
       ev[NX+2] = ev[NX+1];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    if ( step == 1 ) d_Src[IND] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    int i = get_local_id(0) + 2;
    real4 e_v = ev[i];
    real ed = e_v.s0;
    real vx = e_v.s1;
    real vy = e_v.s2;
    real vz = e_v.s3;
    real pressure = P(ed);
    real u0 = gamma(vx, vy, vz);

    // Tzz_tilde = T^{eta eta} * tau^2; no 1/tau in vz
    real Tzz_tilde = (ed + pressure)*u0*u0*vz*vz + pressure;
    real Ttz_tilde = (ed + pressure)*u0*u0*vz;
    real4 src4_christoeffel = {Tzz_tilde, 0.0f, 0.0f, Ttz_tilde};

    d_Src[IND] = d_Src[IND] - src4_christoeffel + kt1d(
           ev[i-2], ev[i-1], e_v, ev[i+1], ev[i+2], tau, ALONG_X)/DX;
}

__kernel void kt_src_alongy(
                     __global real4 * d_Src,     // out put
		     __global real4 * d_ev,
		     const real time,
		     const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    if ( I >= NX || J >= NY || K >= NZ ) return;

    int IND = I*NY*NZ + J*NZ + K;
    // store one line of data in local memory
    __local real4 ev[NY+4];

    // load 1D data to local memory
    for ( int i = get_local_id(1); i < NY; i += get_local_size(1) ) {
        ev[i+2] = d_ev[IND];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    // set boundary condition
    if ( get_local_id(1) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NY+3] = ev[NY+1];
       ev[NY+2] = ev[NY+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_local_id(1) + 2;
    d_Src[IND] = d_Src[IND] + kt1d(ev[i-2], ev[i-1],
	  e_v[i], ev[i+1], ev[i+2], time, ALONG_Y)/DY;
}


__kernel void kt_src_alongz(
                     __global real4 * d_Src,     // out put
		     __global real4 * d_ev,
		     const real time,
		     const int step) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    if ( I >= NX || J >= NY || K >= NZ ) return;
    int IND = I*NY*NZ + J*NZ + K;

    // store one line of data in local memory
    __local real4 ev[NZ+4];

    // load 1D data to local memory
    for ( int i = get_local_id(2); i < NZ; i += get_local_size(2) ) {
        ev[i+2] = d_ev[IND];
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    // set boundary condition
    if ( get_local_id(2) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NZ+3] = ev[NZ+1];
       ev[NZ+2] = ev[NZ+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int i = get_local_id(2) + 2;
    d_Src[IND] = d_Src[IND] + kt1d(ev[i-2], ev[i-1], e_v[i], ev[i+1], ev[i+2],
		                   time, ALONG_Z)/(time*DZ);
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
