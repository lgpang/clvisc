#define REDUCTION_BSZ 256
#define REDUCTION_NBlocks 64

// redefine the float or double as real
#include<real_type.h>

__kernel __attribute__((work_group_size_hint(REDUCTION_BSZ, 1, 1))) 
void reduction_stage1( 
     __global real4 * x,   //input array
     __global real * semi_result, //store the results for first stage reduction
     int size)
{

int gid = get_global_id(0);
int lid = get_local_id(0);

__local real xmax[ REDUCTION_BSZ ];  // REDUCTION_BSZ=Block size

if ( gid < size ) xmax[lid] = x[gid].s0;

while ( (int)(gid+get_global_size(0)) < size ) {
  gid += get_global_size(0);
  real new_x = x[gid].s0;
  if ( new_x > xmax[lid] ) xmax[lid] = new_x;
}

barrier( CLK_LOCAL_MEM_FENCE );

for( int s = get_local_size(0) >> 1; s>0; s >>= 1 ){
     if(lid < s) xmax[ lid ] = (xmax[lid] > xmax[s+lid]) ? xmax[lid] : xmax[s+lid] ;
     barrier( CLK_LOCAL_MEM_FENCE );
}

if( lid == 0 ) semi_result[ get_group_id(0) ] = xmax[ 0 ];

}

// Use one workgroup to reduce REDUCTION_NBlocks elements
__kernel __attribute__((work_group_size_hint(REDUCTION_NBlocks, 1, 1)))
 void reduction_stage2( 
    __global real * x, //store the results for first stage reduction
    __global real * final_result) 
{

int lid = get_local_id(0);

__local real xmax[ REDUCTION_NBlocks ];  // REDUCTION_BSZ=Block size

xmax[lid] = x[ lid ];
barrier( CLK_LOCAL_MEM_FENCE );

for( int s = get_local_size(0) >> 1; s>0; s >>= 1 ){
     if(lid < s) xmax[ lid ] = (xmax[lid] > xmax[s+lid]) ? xmax[lid] : xmax[s+lid] ;
     barrier( CLK_LOCAL_MEM_FENCE );
}

if( lid == 0 ) final_result[0] = xmax[ 0 ];

}
