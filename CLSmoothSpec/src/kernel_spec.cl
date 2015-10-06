#include<helper.h>

/** First stage: calc sub spec for each particle */
__kernel void get_sub_dNdYPtdPtdPhi(  
        __global real8  * d_SF,            
        __global real  * d_SubSpec,            
        __constant real4 * d_HadronInfo,           
        __constant real * d_Y ,           
        __constant real * d_Pt,
        __constant real * d_CPhi,
        __constant real * d_SPhi,
        const int pid
        )
{
    int I = get_global_id(0);

    int tid = get_local_id(0) ;


    __local real subspec[ BSZ ];

    real8 SF;

    real4 HadronInfo = d_HadronInfo[ pid ];
    real mass = HadronInfo.s0;
    real gspin = HadronInfo.s1;
    real fermi_boson = HadronInfo.s2;
    real muB = HadronInfo.s3;
    real dof = gspin /pown( 2.0*M_PI_F, 3 ) ;

    //divide phi with NPHI=48 into 12 real4 types to speed up
    real dNdYPtdPtdPhi[NY][NPT][NPHI];

    for( int k=0; k!=NY; k++ )
        for( int l=0; l!=NPT; l++ )
            for( int m=0; m!=NPHI; m++ ){
                dNdYPtdPtdPhi[k][l][m] = 0.0f;
    }
    
    while( I < SizeSF ){
        SF = d_SF[I];
        real gamma =  1.0/sqrt( max(1.0E-30, 1.0-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6 ) )  ;
        for( int k=0; k!=NY; k++ )
            for( int l=0; l!=NPT; l++ ){
                real pt =  d_Pt[l];
                real mt = sqrt( mass*mass + pt*pt );
                real mtcy =  mt* cosh( d_Y[k] - SF.s7 ) ;
                real mtsy =  mt* sinh( d_Y[k] - SF.s7 ) ;
                for( int m=0; m!=NPHI; m++ ){
                    real  px = pt *  d_CPhi[m];
                    real  py = pt *  d_SPhi[m];
                    real efkt = ( gamma * ( mtcy - px*SF.s4 - py*SF.s5 - mtsy*SF.s6 ) - muB ) / Tfrz;
                    //real efkt = ( gamma * ( mtcy - px*SF.s4 - py*SF.s5 - mtsy*SF.s6 ) ) / Tfrz;
                    real volum = mtcy*SF.s0 - px*SF.s1 - py*SF.s2 - mtsy*SF.s3;
                    dNdYPtdPtdPhi[k][l][m] += dof * volum / ( exp(efkt) + fermi_boson );
                    /** in units of GeV.fm^3 */
                }
        }
        I += get_global_size(0);
    }


    for( int k=0; k!=NY; k++ )
        for( int l=0; l!=NPT; l++ )
            for( int m=0; m!=NPHI; m++ ){
                subspec[tid] = dNdYPtdPtdPhi[k][l][m];
                barrier( CLK_LOCAL_MEM_FENCE );

                //// do reduction in shared mem
                for(unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1) 
                {
                    if(tid < s) 
                    {
                        subspec[tid] += subspec[tid + s];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                ////Dont know why, but the following code give out wrong results
                //if( BSZ >= 512 ){ if( tid < 256 ) subspec[tid] += subspec[ 256 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
                //if( BSZ >= 256 ){ if( tid < 128 ) subspec[tid] += subspec[ 128 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
                //if( BSZ >= 128 ){ if( tid < 64 )  subspec[tid]  += subspec[ 64 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };

                ////// for last warp, don't need mem fence
                //if( tid < 32 ){
                //   if( BSZ >= 64 ) subspec[ tid ] += subspec[ 32 + tid ];
                //   if( BSZ >= 32 ) subspec[ tid ] += subspec[ 16 + tid ];
                //   if( BSZ >= 16 ) subspec[ tid ] += subspec[ 8 + tid ];
                //   if( BSZ >= 8 )  subspec[ tid ] += subspec[ 4 + tid ];
                //   if( BSZ >= 4 )  subspec[ tid ] += subspec[ 2 + tid ];
                //   if( BSZ >= 2 )  subspec[ tid ] += subspec[ 1 + tid ];
                //}

                if(tid == 0) d_SubSpec[ k*NPT*NPHI*NBlocks + l*NPHI*NBlocks + m*NBlocks + get_group_id(0) ] = subspec[ tid ];
            }

}


/** The second stage on sum reduction 
*   For each Y,Pt,Phi we need to sum NBlocks dNdYptdptdphi from the first stage */
__kernel void get_dNdYPtdPtdPhi(  
        __global real  * d_SubSpec,            
        __global real  * d_Spec,
        const int pid)
{
        int I = get_global_id(0);
        int tid = get_local_id(0);
        int localSize = get_local_size(0);

        __local real spec[ NBlocks ];

        spec[ tid ] = d_SubSpec[ I ];

        barrier( CLK_LOCAL_MEM_FENCE );

        for( int s = localSize>>1; s>0; s >>= 1 ){
             if( tid < s ){
                 spec[ tid ] += spec[ s + tid ];
             }
             barrier( CLK_LOCAL_MEM_FENCE );
        }

        /** unroll the last warp because they are synced automatically */
        /** \bug unroll gives out wrong results */
        //if( NBlocks >= 512 ){ if( tid < 256 ) spec[tid] += spec[ 256 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
        //if( NBlocks >= 256 ){ if( tid < 128 ) spec[tid] += spec[ 128 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
        //if( NBlocks >= 128 ){ if( tid < 64 )  spec[tid] += spec[ 64 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };

        //if( tid < 32 ){
        //   if( NBlocks >= 64 ) spec[ tid ] += spec[ 32 + tid ];
        //   if( NBlocks >= 32 ) spec[ tid ] += spec[ 16 + tid ];
        //   if( NBlocks >= 16 ) spec[ tid ] += spec[ 8 + tid ];
        //   if( NBlocks >= 8 )  spec[ tid ] += spec[ 4 + tid ];
        //   if( NBlocks >= 4 )  spec[ tid ] += spec[ 2 + tid ];
        //   if( NBlocks >= 2 )  spec[ tid ] += spec[ 1 + tid ];
        //}

        if ( tid == 0 ) d_Spec[ pid*NY*NPT*NPHI + get_group_id(0) ] = spec[ tid ];

}
