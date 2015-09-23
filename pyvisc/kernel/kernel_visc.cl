#include<helper.h>

#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))

constant int INDEX[4][4] = \
{   0, 1, 2, 3, 
    1, 4, 5, 6,
    2, 5, 7, 8,
    3, 6, 8, 9};  



void CalcSourcesVisc( 
        __private real SPI[10],             \
        __private real Sigma[10],             \
        __local real Ed[BSZ][BSZ][BSZ],     \
        __local real4 Umu0[BSZ][BSZ][BSZ] , \
        __local real4 Umu1[BSZ][BSZ][BSZ] , \
        __private real pisrc[10], real tau, int i, int j, int k);


//////////////////////////////////////////////////////////
/*! */
__kernel void initVisc(  
	__global real * d_pi0,            
	__global real * d_pi1,            
	__global real4 * d_ev0,
	__global real4 * d_ev1,
    const real tau0,
	const int  Size)
{   //In this kernel, globalsize=NDRange( Size )

    int idx = get_global_id(0);

    d_Umu0[ idx ] = d_Umu1[ idx ];

    for(int i=0; i<10; i++) {
       d_pi0[ 10*idx + i ] = 0.0f;
       d_pi1[ 10*idx + i ] = 0.0f;
    }

    real S0 = S( d_Ed[idx] );
    d_pi0[10*idx + 4] = 2.0*ETAOS*S0/(3.0*tau0);
    d_pi0[10*idx + 7] = 2.0*ETAOS*S0/(3.0*tau0);
    d_pi0[10*idx + 9] =-4.0*ETAOS*S0/(3.0*tau0);


//    int I = get_global_id(0);
//    int J = get_global_id(1);
//    int K = get_global_id(2);
//
//    //load Ed Umu to shared momory
//    int i = get_local_id(0) + 2;
//    int j = get_local_id(1) + 2;
//    int k = get_local_id(2) + 2;
//
//    /** halo cells i=0, i=1, BSZ-2, BSZ-1 
//     *  We need 2 halo cells in the right for KT algorithm*/
//    __local real4 Umu0[BSZ][BSZ][BSZ];
//    __local real4 Umu1[BSZ][BSZ][BSZ];
//    __local real    Ed[BSZ][BSZ][BSZ];
//    __local real   CS2[BSZ][BSZ][BSZ];
//
//
//    int IND = I*NY*NZ + J*NZ + K;
//
//    loadEdUmu( d_Umu0, d_Umu1, d_Ed,  Umu0, Umu1, Ed, CS2, i, j, k, IND ); 
//
//    // halo cells for i==0 and i== BSZ-1 and BSZ-2
//    if( get_local_id(0) == 0 ){
//        /// left: for 2 left most cells use constant extrapolation
//        IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-2)*NY*NZ + J*NZ + K );
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, 0, j, k, IND ); 
//
//        IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-1)*NY*NZ + J*NZ + K );
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, 1, j, k, IND ); 
//
//        /// right: for 2 right most cells use constant extrapolation
//        IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-4)*NY*NZ + J*NZ + K );
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, BSZ-2, j, k, IND ); 
//
//        IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-3)*NY*NZ + J*NZ + K );
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, BSZ-1, j, k, IND ); 
//    }
//
//    // halo cells for j==0 and j==BSZ-1
//    if( get_local_id(1) == 0 ){
//        IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-2)*NZ + K);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, 0, k, IND ); 
//        IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-1)*NZ + K);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, 1, k, IND ); 
//        IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-4)*NZ + K);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, BSZ-2, k, IND ); 
//        IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-3)*NZ + K);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, BSZ-1, k, IND ); 
//    }
//    // halo cells for k==0 and k==BSZ-1
//    if( get_local_id(2) == 0 ){
//        IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-2);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, 0, IND ); 
//        IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-1);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, 1, IND ); 
//        IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-4);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, BSZ-2, IND ); 
//        IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-3);
//        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, BSZ-1, IND ); 
//    }
//    barrier( CLK_LOCAL_MEM_FENCE );
//
//    /** src contribution for Tmn */
//    real pi1[10], SPI[10], Sigma[10];
//    for(int n=0; n<10; n++) pi1[n]=d_pi1[ 10*IND + n ] ;
//    //for(int n=0; n<10; n++) pi1[n]=d_pi0[ 10*IND + n ] ;
//    CalcSourcesVisc( SPI, Sigma, Ed, Umu0, Umu1, pi1, tau0, i, j, k );
//
//    real etav  = hbarc * ETAOS * S( Ed[i][j][k] );
//
//    for( int n=0; n<10; n++ ) {
//        d_pi0[ 10*IND + n ] = 0.0;
//        //d_pi0[ 10*IND + n ] = 0.8*etav * Sigma[ n ];
//        //d_pi1[ 10*IND + n ] = etav * Sigma[ n ];
//        d_pi1[ 10*IND + n ] = 0.0;
//    }

}


__kernel void stepUpdateVisc(  
        __global real  * d_pi0,
        __global real  * d_pi1,
        __global real4 * d_Umu0,            
        __global real4 * d_Umu1,            
        __global real  * d_Ed ,           
        __global real  * d_Newpi,
        __global real4 * d_Src,
        const real tau,
        const int  halfStep,
        const int  Size)
{
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    //load Ed Umu to shared momory
    int i = get_local_id(0) + 2;
    int j = get_local_id(1) + 2;
    int k = get_local_id(2) + 2;

    /** halo cells i=0, i=1, BSZ-2, BSZ-1 
     *  We need 2 halo cells in the right for KT algorithm*/
    __local real  pimn[BSZ][BSZ][BSZ];
    __local real4 Umu0[BSZ][BSZ][BSZ];
    __local real4 Umu1[BSZ][BSZ][BSZ];
    __local real    Ed[BSZ][BSZ][BSZ];
    __local real   CS2[BSZ][BSZ][BSZ];


    int IND = I*NY*NZ + J*NZ + K;

    loadEdUmu( d_Umu0, d_Umu1, d_Ed,  Umu0, Umu1, Ed, CS2, i, j, k, IND ); 

    // halo cells for i==0 and i== BSZ-1 and BSZ-2
    if( get_local_id(0) == 0 ){
        /// left: for 2 left most cells use constant extrapolation
        IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-2)*NY*NZ + J*NZ + K );
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, 0, j, k, IND ); 

        IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-1)*NY*NZ + J*NZ + K );
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, 1, j, k, IND ); 

        /// right: for 2 right most cells use constant extrapolation
        IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-4)*NY*NZ + J*NZ + K );
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, BSZ-2, j, k, IND ); 

        IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-3)*NY*NZ + J*NZ + K );
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, BSZ-1, j, k, IND ); 
    }

    // halo cells for j==0 and j==BSZ-1
    if( get_local_id(1) == 0 ){
        IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-2)*NZ + K);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, 0, k, IND ); 
        IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-1)*NZ + K);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, 1, k, IND ); 
        IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-4)*NZ + K);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, BSZ-2, k, IND ); 
        IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-3)*NZ + K);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, BSZ-1, k, IND ); 
    }
    // halo cells for k==0 and k==BSZ-1
    if( get_local_id(2) == 0 ){
        IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-2);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, 0, IND ); 
        IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-1);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, 1, IND ); 
        IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-4);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, BSZ-2, IND ); 
        IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-3);
        loadEdUmu( d_Umu0, d_Umu1, d_Ed, Umu0, Umu1, Ed, CS2, i, j, BSZ-1, IND ); 
    }
    barrier( CLK_LOCAL_MEM_FENCE );

    /** src contribution for Tmn */
    real pi1[10], SPI[10], Sigma[10];

    for(int n=0; n<10; n++) pi1[n]=d_pi1[ 10*IND + n ] ;
    //for(int n=0; n<10; n++) pi1[n]=d_pi0[ 10*IND + n ] ;

    CalcSourcesVisc( SPI, Sigma, Ed, Umu0, Umu1, pi1, tau + (1.0-halfStep)*DT, i, j, k );


    real etav  = ETAOS * S( Ed[i][j][k] );

    /** \hat{eta_v}, \hat{tau_pi} all of them are dimensionless*/
    //real etavH = ETAOS * S( Ed[i][j][k] ) * pow( Ed[i][j][k], -0.75 );
    real etavH = ETAOS * 4.0 / 3.0;
    real taupiH = LAM1H*LAM1H*etavH / 3.0 ;
    real one_o_taupi = 1.0 / ( taupiH * pow( Ed[i][j][k], -0.25 ) );

    //real one_o_taupi = T( Ed[i][j][k] ) /( 3.0*fmax( acu, ETAOS )*hbarc );


    real4 src_for_tmn;
    for( int mu = 0 ; mu < 4; mu ++ )
        for( int nu=mu; nu < 4; nu ++ ){
            IND = I*NY*NZ + J*NZ + K;
            //loadpimn( d_pi0, pimn, i, j,  k, IND, mu, nu );
            pimn[i][j][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
            if( get_local_id(0) == 0 ){
                /// left: for 2 left most cells use constant extrapolation
                IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-2)*NY*NZ + J*NZ + K );
                pimn[0][j][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(0)==0) ? (0*NY*NZ+J*NZ+K) : ( (I-1)*NY*NZ + J*NZ + K );
                pimn[1][j][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                /// right: for 2 right most cells use constant extrapolation
                IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-4)*NY*NZ + J*NZ + K );
                pimn[BSZ-2][j][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(0)==get_num_groups(0)-1) ? ( (NX-1)*NY*NZ+J*NZ+K ): ( (I+BSZ-3)*NY*NZ + J*NZ + K );
                pimn[BSZ-1][j][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
            }
            if( get_local_id(1) == 0 ){
                IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-2)*NZ + K);
                pimn[i][0][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(1)==0) ? (I*NY*NZ + 0*NZ + K) : (I*NY*NZ + (J-1)*NZ + K);
                pimn[i][1][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-4)*NZ + K);
                pimn[i][BSZ-2][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(1)==get_num_groups(1)-1) ? (I*NY*NZ + (NY-1)*NZ + K) : (I*NY*NZ + (J+BSZ-3)*NZ + K);
                pimn[i][BSZ-1][k]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
            }

            // halo cells for k==0 and k==BSZ-1
            if( get_local_id(2) == 0 ){
                IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-2);
                pimn[i][j][0]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(2)==0) ? (I*NY*NZ + J*NZ + 0) : (I*NY*NZ + J*NZ + K-1);
                pimn[i][j][1]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-4);
                pimn[i][j][BSZ-2]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
                IND = (get_group_id(2)==get_num_groups(2)-1) ? (I*NY*NZ + J*NZ + NZ-1) : (I*NY*NZ + J*NZ + K+BSZ-3);
                pimn[i][j][BSZ-1]  =  d_pi0[ 10*IND + INDEX[mu][nu] ];
            }
            barrier( CLK_LOCAL_MEM_FENCE );


            IND = I*NY*NZ + J*NZ + K;
            //real pisrc[10] = vload10( IND*size_of(real), d_pi1 );
           //real src = CalcSources( Ed, Umu0, Umu1, pisrc, time, i, j, k, mu, nu );
            real src = SPI[ INDEX[mu][nu] ];
            //Use KT algorithm to update

            real tmp = KT3D( pimn, Umu1, CS2, tau, i, j, k, I, J, K ) + DT*src ;
            //real tmp = Upwind( pimn, Umu1, CS2, tau, i, j, k, I, J, K ) + (1.0-0.5*halfStep)*DT*src ;

            //real piNS = etav * Sigma[INDEX[mu][nu]];
            real piNS = Sigma[INDEX[mu][nu]];

            //if( one_o_taupi/Umu1[i][j][k].s0 < 0.5 ){
              d_Newpi[10*IND + INDEX[mu][nu]] = ( tmp - piNS )*exp( -one_o_taupi*DT/Umu1[i][j][k].s0 ) + piNS;
            //}
            //else{
            //  d_Newpi[10*IND + INDEX[mu][nu]] =  tmp + (piNS-tmp)*one_o_taupi /Umu1[i][j][k].s0*(1.0-0.5*halfStep)*DT;
            //}

            if( mu==0 && nu==0 ) src_for_tmn.s0 = src + (piNS-tmp)*one_o_taupi /Umu1[i][j][k].s0;
            if( mu==0 && nu==1 ) src_for_tmn.s1 = src + (piNS-tmp)*one_o_taupi /Umu1[i][j][k].s0;
            if( mu==0 && nu==2 ) src_for_tmn.s2 = src + (piNS-tmp)*one_o_taupi /Umu1[i][j][k].s0;
            if( mu==0 && nu==3 ) src_for_tmn.s3 = src + (piNS-tmp)*one_o_taupi /Umu1[i][j][k].s0;
        }

        IND = I*NY*NZ + J*NZ + K ;
        d_Src[ IND ] = (tau + (1.0-halfStep)*DT) * src_for_tmn ;   // if halfStep==1, time=tau

}

/** \partial_{mu} A4 */
real4 Partial(int mu, __local real4 A0[BSZ][BSZ][BSZ], __local real4 A1[BSZ][BSZ][BSZ], int i, int j, int k)
{
    //enum {t, x, y, etas}; //here t means propertime INDEX
    real4 result=(real4) (0.0, 0.0, 0.0, 0.0);
    real4 fb, fc, ff;
    switch(mu){
	case 0:
        /** \todo to see what's the maximum DtDU */
	    result = (A1[i][j][k]-A0[i][j][k])/DT ;
	    break;
	case 1: 
        fb = theta*(A1[i][j][k]-A1[i-1][j][k]);
        fc = 0.5*(A1[i+1][j][k]-A1[i-1][j][k]);
        ff = theta*(A1[i+1][j][k]-A1[i][j][k]);
	    result = minmod4( &fb, &fc, &ff )/DX;
	    break;

	case 2:	
        fb = theta*(A1[i][j][k]-A1[i][j-1][k]);
        fc = 0.5*(A1[i][j+1][k]-A1[i][j-1][k]);
        ff = theta*(A1[i][j+1][k]-A1[i][j][k]);
	    result = minmod4( &fb, &fc, &ff )/DY;
	    break;

	case 3:	
        fb = theta*(A1[i][j][k]-A1[i][j][k-1]);
        fc = 0.5*(A1[i][j][k+1]-A1[i][j][k-1]);
        ff = theta*(A1[i][j][k+1]-A1[i][j][k]);
	    result = minmod4( &fb, &fc, &ff )/DZ;
	    break;
	default:
	    break;	
    }
    return result;
}

void Calc_NablaU( __private real  NablaU[4][4],
                  __private real  PIVI[4],
                  __local real4 Umu0[BSZ][BSZ][BSZ],
                  __local real4 Umu1[BSZ][BSZ][BSZ],
                  int i, int j, int k, real tau )
{
    real4 PtUmu =  Partial( 0, Umu0, Umu1, i, j, k);
    real4 PxUmu =  Partial( 1, Umu0, Umu1, i, j, k);
    real4 PyUmu =  Partial( 2, Umu0, Umu1, i, j, k);
    real4 PzUmu =  Partial( 3, Umu0, Umu1, i, j, k);

    /** \bug the pivi calculation has problem here */
    //PIVI[0] = 0.0;
    //PIVI[1] = (PxUmu.s1 - Umu1[i][j][k].s1/Umu1[i][j][k].s0 * PxUmu.s0 ) / Umu1[i][j][k].s0;
    //PIVI[2] = (PyUmu.s2 - Umu1[i][j][k].s2/Umu1[i][j][k].s0 * PyUmu.s0 ) / Umu1[i][j][k].s0;
    //PIVI[3] = (PzUmu.s3 - Umu1[i][j][k].s3/Umu1[i][j][k].s0 * PzUmu.s0 ) / Umu1[i][j][k].s0;

    ///** \partial_x Vx */
    real vxm1 = Umu1[i-1][j][k].s1 / Umu1[i-1][j][k].s0 ;
    real vxp1 = Umu1[i+1][j][k].s1 / Umu1[i+1][j][k].s0 ;
    real vx   = Umu1[i  ][j][k].s1 / Umu1[i  ][j][k].s0 ;
    PIVI[1]   = minmod( theta*(vx - vxm1), 0.5*(vxp1 - vxm1), theta*(vxp1 - vx))/DX;

    /** \partial_y Vy */
    real vym1 = Umu1[i][j-1][k].s2 / Umu1[i][j-1][k].s0 ;
    real vyp1 = Umu1[i][j+1][k].s2 / Umu1[i][j+1][k].s0 ;
    real vy   = Umu1[i][j  ][k].s2 / Umu1[i][j  ][k].s0 ;
    PIVI[2]   = minmod( theta*(vy - vym1), 0.5*(vyp1 - vym1), theta*(vyp1 - vy))/DY;

    /** \partial_\eta V\eta */
    real vzm1 = Umu1[i][j][k-1].s3 / Umu1[i][j][k-1].s0 ;
    real vzp1 = Umu1[i][j][k+1].s3 / Umu1[i][j][k+1].s0 ;
    real vz   = Umu1[i][j][k  ].s3 / Umu1[i][j][k  ].s0 ;
    PIVI[3]   = minmod( theta*(vz - vzm1), 0.5*(vzp1 - vzm1), theta*(vzp1 - vz))/DY;

    NablaU[0][0] = PtUmu.s0;
    NablaU[0][1] = PtUmu.s1;
    NablaU[0][2] = PtUmu.s2;
    NablaU[0][3] = PtUmu.s3 + Umu1[i][j][k].s3 / tau;

    NablaU[1][0] = PxUmu.s0;
    NablaU[1][1] = PxUmu.s1;
    NablaU[1][2] = PxUmu.s2;
    NablaU[1][3] = PxUmu.s3;

    NablaU[2][0] = PyUmu.s0;
    NablaU[2][1] = PyUmu.s1;
    NablaU[2][2] = PyUmu.s2;
    NablaU[2][3] = PyUmu.s3;

    NablaU[3][0] = PzUmu.s0 + Umu1[i][j][k].s3 * tau;
    NablaU[3][1] = PzUmu.s1;
    NablaU[3][2] = PzUmu.s2;
    NablaU[3][3] = PzUmu.s3 + Umu1[i][j][k].s0 / tau;

}

/** \pi^{< mu}_{lam} * pi^{nu> lam} self coupling term **/

inline real PiPi( int lam, int mu, int nu, __private real pimn[10], real Delta[4][4],  real tau )
{
    /** pi_{\mu}^{lam} **/
    real4 A = (real4) ( pimn[ INDEX[0][lam] ], -pimn[ INDEX[1][lam] ], -pimn[ INDEX[2][lam] ], -tau*tau*pimn[ INDEX[3][lam] ] );
    real4 Delta4[] = {(real4) ( Delta[0][0], Delta[0][1], Delta[0][2], Delta[0][3] ), 
                      (real4) ( Delta[1][0], Delta[1][1], Delta[1][2], Delta[1][3] ), 
                      (real4) ( Delta[2][0], Delta[2][1], Delta[2][2], Delta[2][3] ), 
                      (real4) ( Delta[3][0], Delta[3][1], Delta[3][2], Delta[3][3] ) }; 

    // Delta^{mu alpha} Delta^{nu beta} * A_{alpha} * A_{beta} 
    real firstTerm = dot( Delta4[mu], A ) * dot( Delta4[nu], A ) ;

    real4 temp = (real4) ( dot( Delta4[0], A ), dot( Delta4[1],A ), dot( Delta4[2],A), dot( Delta4[3], A ) );

    // -1/3.0 * Delta[mu][nu]  Delta^{alpha beta} * A_{alpha} * A_{beta} 
    real secondTerm = -1.0/3.0 * Delta[mu][nu] * dot( temp, A );

    return firstTerm + secondTerm;
}


/** \todo Calc all the source terms here to reduce repeated calculations */
void CalcSourcesVisc( 
        __private real SPI[10],             \
        __private real Sigma[10],             \
        __local real Ed[BSZ][BSZ][BSZ],     \
        __local real4 Umu0[BSZ][BSZ][BSZ] , \
        __local real4 Umu1[BSZ][BSZ][BSZ] , \
        __private real pisrc[10], real tau, int i, int j, int k)
{
        real gu[4][4] = {0.0};
        real Gamma[4][4][4] = {0.0};
        real Delta[4][4];

        // Notice for BoWen Ini, [Ed] = fm^{-4}
        //real coef_pipi = 2.0 / sqrt( 4.0*etav/3.0 * one_o_taupi );
        //real etavH = ETAOS * S( Ed[i][j][k] ) * pow( Ed[i][j][k], -0.75 );
        real etavH = ETAOS * 4.0 / 3.0 ;
        real taupiH = LAM1H*LAM1H*etavH / 3.0 ;
        real one_o_taupi = 1.0 / ( taupiH * pow( Ed[i][j][k], -0.25 ) );
        real etav  = ETAOS * S( Ed[i][j][k] );

        //real coef_pipi = LAM1H / Ed[i][j][k] * one_o_taupi ;
        real coef_pipi = LAM1H / Ed[i][j][k] ;

        gu[0][0] = 1.0;
        gu[1][1] =-1.0;
        gu[2][2] =-1.0;
        gu[3][3] =-1.0/(tau * tau);

        Gamma[3][3][0] = 1.0/tau;
        Gamma[3][0][3] = 1.0/tau;
        Gamma[0][3][3] = tau;

        real NablaU[4][4];
        real PIVI[4];
        Calc_NablaU( NablaU, PIVI, Umu0, Umu1, i, j, k, tau );

        real DU[4] ;
        real Umu[4] = {Umu1[i][j][k].s0 , Umu1[i][j][k].s1, Umu1[i][j][k].s2, Umu1[i][j][k].s3 };
        for(int n=0; n<4; n++){
           DU[n] =  Umu[0] * NablaU[0][n] + \
                Umu[1] * NablaU[1][n] + \
                Umu[2] * NablaU[2][n] + \
                Umu[3] * NablaU[3][n] ;
        }

        real Theta = NablaU[0][0] + NablaU[1][1] + NablaU[2][2] + NablaU[3][3];

        for( int mu=0; mu<4; mu++)
        for( int nu=0; nu<4; nu++){
            Delta[mu][nu] = gu[mu][nu] - Umu[mu]*Umu[nu];
        }

        for( int mu=0; mu<4; mu++)
        for( int nu=mu; nu<4; nu++){

        real DUU = Umu[mu] * DU[nu] + Umu[nu] * DU[mu] ;

        /**  Theta = Nabla cdot U  */


        Sigma[ INDEX[mu][nu] ] = etav * ( gu[mu][mu] * NablaU[mu][nu] + gu[nu][nu]*NablaU[nu][mu] \
                     - DUU - 2.0/3.0*Theta * Delta[mu][nu] );
         
        real B = -4.0/3.0*pisrc[ INDEX[mu][nu] ] * Theta;

        //for( int a=0; a!=4; a++ ){
        //     B += -gd[a][a] * ( Umu[mu] * pisrc[ INDEX[nu][a] ] \
        //                      + Umu[nu] * pisrc[ INDEX[mu][a] ] ) \
        //                      * DU[a];
        //}

        B += - ( Umu[mu] * pisrc[ INDEX[nu][0] ] + Umu[nu] * pisrc[ INDEX[mu][0] ] ) * DU[0]\
         +  ( Umu[mu] * pisrc[ INDEX[nu][1] ] + Umu[nu] * pisrc[ INDEX[mu][1] ] ) * DU[1]\
         +  ( Umu[mu] * pisrc[ INDEX[nu][2] ] + Umu[nu] * pisrc[ INDEX[mu][2] ] ) * DU[2]\
         + tau*tau * ( Umu[mu] * pisrc[ INDEX[nu][3] ] + Umu[nu] * pisrc[ INDEX[mu][3] ] ) * DU[3];


        /** pi^{<mu}_{lam} pi^{nu> lam} */
        //B -= coef_pipi*one_o_taupi*( PiPi( 0, mu, nu, pisrc, Delta, tau )
        //    -PiPi( 1, mu, nu, pisrc, Delta, tau )
        //    -PiPi( 2, mu, nu, pisrc, Delta, tau )
        //    -PiPi( 3, mu, nu, pisrc, Delta, tau )*tau*tau);

        //Sigma[ INDEX[mu][nu] ] -= coef_pipi*( \
        //     PiPi( 0, mu, nu, pisrc, Delta, tau )
        //    -PiPi( 1, mu, nu, pisrc, Delta, tau )
        //    -PiPi( 2, mu, nu, pisrc, Delta, tau )
        //    -PiPi( 3, mu, nu, pisrc, Delta, tau )*tau*tau);



        int n = INDEX[mu][nu];
        SPI[n] = B/Umu[0];
        SPI[n] += pisrc[n] * ( PIVI[1] + PIVI[2] + PIVI[3] );
        //for( int l=0; l<4; l++ ){
        //    SPI[n] += -( Gamma[mu][l][0] * pisrc[ INDEX[l][nu] ] + Gamma[nu][l][0]*pisrc[INDEX[mu][l]] );
        //    SPI[n] += -Umu[3]/Umu[0]*( Gamma[mu][l][3] * pisrc[ INDEX[l][nu] ] + Gamma[nu][l][3]*pisrc[INDEX[mu][l]] );
        //}

        SPI[n] +=  -Umu[3]/Umu[0]*( Gamma[mu][0][3] * pisrc[ INDEX[0][nu] ] + Gamma[nu][0][3]*pisrc[INDEX[mu][0]] ) \
        -( Gamma[mu][3][0] * pisrc[ INDEX[3][nu] ] + Gamma[nu][3][0]*pisrc[INDEX[mu][3]] ) \
        -Umu[3]/Umu[0]*( Gamma[mu][3][3] * pisrc[ INDEX[3][nu] ] + Gamma[nu][3][3]*pisrc[INDEX[mu][3]] );
        }
}

#define maxPiRatio 1
#define smallness  0.01
#define accuracy   0.01
#define ms         1.0

__kernel void updateGlobalMemVisc(
    __global real * d_pi0, 
    __global real * d_pi1, 
    __global real4 * d_Umu0, 
    __global real4 * d_Umu1, 
    __global real * d_Newpi, 
    __global real * d_Ed, 
    const  real   tau,
    const    int  halfStep,
    const    int  Size )
{

    int IND = get_global_id(0);

    real4 Umu = d_Umu1[IND];

    d_Umu0[IND] = Umu;

    real pimn[10];

    for(int i=0; i<10; i++){
       pimn[i] = 0.5*( d_pi0[ 10*IND + i ] + d_pi1[ 10*IND+i ] );
    }
    real time = tau + DT;

    real4 V = Umu / Umu.s0; 


//    pimn[1] = V.s1 * pimn[4] + V.s2 * pimn[5] + time*time*V.s3*pimn[6];
//    pimn[2] = V.s1 * pimn[5] + V.s2 * pimn[7] + time*time*V.s3*pimn[8];
//    pimn[3] = V.s1 * pimn[6] + V.s2 * pimn[8] + time*time*V.s3*pimn[9];
//    pimn[0] = pimn[4] + pimn[7] + time*time*pimn[9];

    int I = IND/(NY*NZ);
    int J = (IND-I*NY*NZ)/NZ;
    int K = (IND - I*NY*NZ - J*NZ );

    real Ed = d_Ed[IND];
    real Pl = P( Ed );
    bool failed = false;
    int Nreg = 0;
    real traceAnormaly = fabs( Ed - 3.0*Pl ) * maxPiRatio;
    real trace_pi = pimn[0] - pimn[4] - pimn[7] - time*time*pimn[9] ;

    if( fabs( trace_pi ) > max( traceAnormaly*accuracy, smallness ) ) {
        failed = true;
        Nreg ++;
    }

    real trans1 = pimn[1] - V.s1 * pimn[4] + V.s2 * pimn[5] + time*time*V.s3*pimn[6];
    if( fabs( trans1 ) > max( accuracy, smallness ) ) {
        failed = true;
        Nreg ++;
    }

    real trans2 = pimn[2] - V.s1 * pimn[5] - V.s2 * pimn[7] - time*time*V.s3*pimn[8];
    if( fabs( trans2 ) > max( accuracy, smallness ) ) {
        failed = true;
        Nreg ++;
    }

    real trans3 = pimn[3] - V.s1 * pimn[6] - V.s2 * pimn[8] - time*time*V.s3*pimn[9];
    if( fabs( trans3 ) > max( accuracy, smallness ) ) {
        failed = true;
        Nreg ++;
    }

    real trans0 = pimn[0] - V.s1 * pimn[1] - V.s2 * pimn[2] - time*time*V.s3*pimn[3];
    if( fabs( trans0 ) > max( accuracy, smallness ) ) {
        failed = true;
        Nreg ++;
    }


    real TrPi2 = pimn[0]*pimn[0] + pimn[4]*pimn[4] + pimn[7]*pimn[7]  \
               + pimn[9]*pimn[9] * pown( time, 4 )                    \
               - 2.0 * pimn[1]* pimn[1] - 2.0*pimn[2]*pimn[2]         \
               - 2.0*pimn[3]*pimn[3]*time*time                        \
               + 2.0*pimn[5]*pimn[5]                                  \
               + 2.0*pimn[6]*pimn[6]*time*time                        \
               + 2.0*pimn[8]*pimn[8]*time*time;                      

    real zero = 0.01 / ( Nreg + 1.0 );
    real ezero = maxPiRatio * fabs( Ed )* zero ;
    real traceAnormaly1 = maxPiRatio * min( max( fabs(Ed-3.0*Pl), 0.001 * Ed ), \
                                           Pl ) * zero;

    if( failed == true ) {
        real maxPi = maxPiRatio * ( Ed + Pl ) ;
        
        real regStrength = max( fabs(trace_pi)/maxPi/ms, 0.00001 );

        regStrength = max( fabs(trans1)/ezero/ms, regStrength );
        regStrength = max( fabs(trans2)/ezero/ms, regStrength );
        regStrength = max( fabs(trans3)/ezero/ms, regStrength );
        regStrength = max( fabs(trans0)/ezero/ms, regStrength );

        regStrength = max( sqrt(fabs(TrPi2))/maxPi, regStrength );

        for(int i=0; i<10; i++){
           pimn[i] *=  tanh( regStrength ) / regStrength ;
        }

        //d_Ed[ IND ] = tanh( regStrength ) / regStrength ; 

        //pimn[1] = V.s1 * pimn[4] + V.s2 * pimn[5] + time*time*V.s3*pimn[6];
        //pimn[2] = V.s1 * pimn[5] + V.s2 * pimn[7] + time*time*V.s3*pimn[8];
        //pimn[3] = V.s1 * pimn[6] + V.s2 * pimn[8] + time*time*V.s3*pimn[9];
        //pimn[0] = pimn[4] + pimn[7] + time*time*pimn[9];

    }

    for(int i=0; i<10; i++){
        d_pi0[ 10*IND+i ] = pimn[i];
        d_pi1[ 10*IND+i ] = pimn[i];
    }

}
