#include<helper.h>

/* Author: LongGang Pang, 2015 
The idea of 3D hypersuface calculation in 4D time-space is based on
2D surface in 3D space. 

For curved surface where not all intersections are on the same plane,
these intersections can be connected by lines in all possible pairs,
then every 3 closed lines (2-simplex) form one piece of hyper surface. However,
not all hyper surfaces are effective hyper surface. Only thoes hyper  
surfaces on the convex hull are needed. For those hyper surface not on the 
convex hull, we remove them if there are points on both
side of this piece of hyper surface.

With all the hyper surfaces on the convex hull, we select half of them
by projecting their norm vector on the low energy density vector.

For 2d hyper surface, if there are only 3 points which form one triangle,
its area can be calculated directly. 

If there are more than 3 points in the same plane, random tiny movement can
be applied to all the points such that the above algorithm can be used.

For 3D hyper surface, 3-simplex is one piece of hyper surface. Everything
is the same as in 2D.
*/

typedef struct {
    real4 p[5];
} simplex;

typedef struct {
    real4 p[4];     // 4 points in 3d hypersf
    real  center;     // surface center 
    real4 norm;     // out ward norm vector
} hypersf;

// get the ourward norm vector of one hyper surface
// mass_center is the center for all intersections
// vector_out = surf_center - mass_center
// norm_out * vout > 0
void get_norm_out(hypersf & surf, real4 mass_center) {
    //real4 surf_center = 0.25f*(surf.p[0]+surf.p[1]+surf.p[2]+surf.p[3]);
    real4 vector_out = surf.center - mass_center;
    // the 3 vector that spans the hypersf
    real4 a = surf.p[1] - surf.p[0];
    real4 b = surf.p[2] - surf.p[0];
    real4 c = surf.p[3] - surf.p[0];
    // norm_vector has 2 directions
    real4 norm_vector = (real4)(
		 a.s1*(b.s2*c.s3-b.s3*c.s2) + a.s2*(b.s3*c.s1-b.s1*c.s3) + a.s3*(b.s1*c.s2-b.s2*c.s1),
		 -(a.s0*(b.s2*c.s3-b.s3*c.s2) + a.s2*(b.s3*c.s0-b.s0*c.s3) + a.s3*(b.s0*c.s2-b.s2*c.s0)),
		a.s0*(b.s1*c.s3-b.s3*c.s1) + a.s1*(b.s3*c.s0-b.s0*c.s3) + a.s3*(b.s0*c.s1-b.s1*c.s0),
		-(a.s0*(b.s1*c.s2-b.s2*c.s1) + a.s1*(b.s2*c.s0-b.s0*c.s2) + a.s2*(b.s0*c.s1-b.s1*c.s0)));

    real projection = dot(vector_out, norm_vector);
    if ( projection < 0 ) norm_vector = - norm_vector;
    return norm_vector;
}

// judge if all the 5 intersections (in one simplex) are coplanar 
inline bool is_coplanar(simplex A){
}

// do tiny move such that they are not coplanar any more
void tiny_move(hypersf & surf){
}

// check if there are points on both side of A, if not return true
// n is the number of intersections
bool is_on_convex_hull(hypersf A, __private real4 * all_points, int n){
}

// get the energy flow vector
real4 energy_flow(__private real4[16] ed_cube) {
}

// select half of the hyper surface with norm*energy_flow>0,
// calc the area for all the selected hypersf on the convex hull
real4 calc_area(simplex A) {
}

// return the index in the global array
// I, J, K: thread id along x, y, z
// i, j, k: 3d pos in d_ev[] array
inline int idn(int I, int J, int K) {
    int i = I*nx_skip;
    int j = J*ny_skip;
    int k = K*nz_skip;
    return i*NY*NZ + j*NZ + k;
}


// output: global d_hypersf array;
__kernel void extract_hypersf(
                       __global real4 * d_evold,
		               __global real4 * d_evnew,
		               const int time_step) 
{
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);

    bool in_range = (I < NX && J < NY && K < NZ);

    if ( in_range ) {
    }
}
