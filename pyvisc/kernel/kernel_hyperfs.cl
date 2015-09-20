#include<helper.h>

/* Author: LongGang Pang, 2015 
The idea of 3D hypersuface calculation in 4D time-space is based on
2D surface in 3D space. 

For curved surface where not all intersections are on the same plane,
these intersections can be connected by lines in all possible pairs,
then every 3 closed lines (2-simplex) form one piece of hyper surface. However,
not all hyper surfaces are effective hyper surface. Only thoes hyper  
surfaces on the convex hull are needed. For those hyper surface not on the convex hull,
we remove them if there are points lies outside the convex hull.

With all the hyper surfaces on the convex hull, we select half of them
by projecting their norm vector on the low energy density vector.

For 2d hyper surface, if there are only 3 points which form one triangle,
its area can be calculated directly. 

If there are more than 3 points in the same plane, random tiny movement can
be applied to all the points such that the above algorithm can be used.

For 3D hyper surface, 3-simplex is one piece of hyper surface. Everything
is the same as in 2D.
*/

constant real4 cube[16] = {
    // tau = tau_old, 8 corners cube
    (real4)(0.0f, 0.0f, 0.0f, 0.0f),
    (real4)(0.0f, 1.0f, 0.0f, 0.0f),
    (real4)(0.0f, 1.0f, 1.0f, 0.0f),
    (real4)(0.0f, 0.0f, 1.0f, 0.0f),
    
    (real4)(0.0f, 0.0f, 0.0f, 1.0f),
    (real4)(0.0f, 1.0f, 0.0f, 1.0f),
    (real4)(0.0f, 1.0f, 1.0f, 1.0f),
    (real4)(0.0f, 0.0f, 1.0f, 1.0f),
    // tau = tau_new, 8 corners cube
    (real4)(1.0f, 0.0f, 0.0f, 0.0f),
    (real4)(1.0f, 1.0f, 0.0f, 0.0f),
    (real4)(1.0f, 1.0f, 1.0f, 0.0f),
    (real4)(1.0f, 0.0f, 1.0f, 0.0f),
    
    (real4)(1.0f, 0.0f, 0.0f, 1.0f),
    (real4)(1.0f, 1.0f, 0.0f, 1.0f),
    (real4)(1.0f, 1.0f, 1.0f, 1.0f),
    (real4)(1.0f, 0.0f, 1.0f, 1.0f)
};

typedef struct {
    real4 p[4];       // 4 points in 3d hypersf
    real4 center;     // surface center 
    real4 norm;       // outward norm vector
} hypersf;

hypersf construct_hypersf(real4 p0, real4 p1, real4 p2, real4 p3,
                          real4 mass_center);

int rand(int* seed);

void tiny_move(hypersf surf, real4 * pnew, int seed_from_gid);

void contribution_from(__private real ed_cube[16], int n, int i, int j, int k,
                   real4 *vl, real4 *vh, real *elsum, real *ehsum);

real4 energy_flow(__private real ed_cube[16]);

real4 calc_area(__private real4 *all_points, real4 energy_flow,
                int num_points);
void get_all_intersections(__private real ed[16],
               __private real4 all_ints[32],
               __private int num_points[1]);

// get the ourward norm vector of one hyper surface
// mass_center is the center for all intersections
// vector_out = surf_center - mass_center
// norm_out * vector_out > 0
hypersf construct_hypersf(real4 p0, real4 p1, real4 p2, real4 p3,
                          real4 mass_center) {
    hypersf surf;
    surf.p[0] = p0;
    surf.p[1] = p1;
    surf.p[2] = p2;
    surf.p[3] = p3;
    surf.center = 0.25f*(p0+p1+p2+p3);
    real4 vector_out = surf.center - mass_center;
    // the 3 vector that spans the hypersf
    real4 a = p1 - p0;
    real4 b = p2 - p0;
    real4 c = p3 - p0;
    // norm_vector has 2 directions
    real4 norm_vector = (real4)(
		 a.s1*(b.s2*c.s3-b.s3*c.s2) + a.s2*(b.s3*c.s1-b.s1*c.s3) + a.s3*(b.s1*c.s2-b.s2*c.s1),
		 -(a.s0*(b.s2*c.s3-b.s3*c.s2) + a.s2*(b.s3*c.s0-b.s0*c.s3) + a.s3*(b.s0*c.s2-b.s2*c.s0)),
		a.s0*(b.s1*c.s3-b.s3*c.s1) + a.s1*(b.s3*c.s0-b.s0*c.s3) + a.s3*(b.s0*c.s1-b.s1*c.s0),
		-(a.s0*(b.s1*c.s2-b.s2*c.s1) + a.s1*(b.s2*c.s0-b.s0*c.s2) + a.s2*(b.s0*c.s1-b.s1*c.s0)));

    real projection = dot(vector_out, norm_vector);
    if ( projection < 0 ) norm_vector = - norm_vector;
    surf.norm = norm_vector;
    return surf;
}

// judge if the point 'pnew' is coplanar with the given hypersf
inline bool is_coplanar(hypersf sf, real4 pnew){
    real4 test_vector = pnew - sf.center;
    bool coplanar = false;
    if ( dot(test_vector, sf.norm) < acu*acu ) {
        coplanar = true;
    }
    return coplanar;
}

// very simple random number generator to provid some random tiny shift
// in case 5 points are coplanar.
int rand(int* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1
    *seed = ((long)(*seed * a))%m;
    return(*seed);
}

// do tiny move such that they are not coplanar any more
void tiny_move(hypersf surf, real4 * pnew, int seed_from_gid){
    int seed = seed_from_gid;
    real maxi = 2147483647.0f;
    while ( is_coplanar(surf, *pnew) ) {
        real4 tiny_shift = (real4)(rand(&seed)/maxi*acu, rand(&seed)/maxi*acu,
                                   rand(&seed)/maxi*acu, rand(&seed)/maxi*acu);
        *pnew = *pnew + tiny_shift;
    }
}

// check if there are points beyond sf, if not return true
// n is the number of intersections
bool is_on_convex_hull(hypersf sf, real4 mass_center, __private real4 * all_points,
                       int num_points, int i, int j, int k, int l){
    real4 outward_vector = sf.center - mass_center;
    for ( int n = 0; n < num_points; n++ ) {
        real4 point = all_points[n];
        if ( n != i && n != j && n != k && n != l ) {
            if ( is_coplanar(sf, point) ) tiny_move(sf, & point, 2355553);
            all_points[n] = point;           // update after tiny move
            real4 test_vector = point - sf.center;
            // if there are points beyond sf, sf is not on convex
            if ( dot(test_vector, outward_vector) > 0.0f ) {
                return false;
            }
        }
    }
    return true;
}

// get the weight of energy from each corner of the cube
void contribution_from(__private real ed_cube[16], int n, int i, int j, int k,
                              real4 *vl, real4 *vh, real *elsum, real *ehsum) {
    int id = 8*n + 4*k + 2*j + i;
    real vl_tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    real vh_tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    real dek = EFRZ - ed_cube[id];
    real adek = fabs(dek);
    if ( dek > 0.0f ) {
        *elsum += adek;
        if ( n == 1 ) vl_tmp[0] = adek;
        if ( i == 1 ) vl_tmp[1] = adek;
        if ( j == 1 ) vl_tmp[2] = adek;
        if ( k == 1 ) vl_tmp[3] = adek;
    } else {
        *ehsum += adek;
        if ( n == 1 ) vh_tmp[0] = adek;
        if ( i == 1 ) vh_tmp[1] = adek;
        if ( j == 1 ) vh_tmp[2] = adek;
        if ( k == 1 ) vh_tmp[3] = adek;
    }
    *vl +=  (real4)(vl_tmp[0], vl_tmp[1], vl_tmp[2], vl_tmp[3]);
    *vh +=  (real4)(vh_tmp[0], vh_tmp[1], vh_tmp[2], vh_tmp[3]);
}

// get the energy flow vector
real4 energy_flow(__private real ed_cube[16]) {
    // vl, vh wight to low/high energy density
    real4 vl = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
    real4 vh = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
    real  elsum = 0.0f;
    real  ehsum = 0.0f;    // sum of ed difference
    // n==0 tau_old
    contribution_from(ed_cube, 0, 0, 0, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 0, 0, 1, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 0, 1, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 0, 1, 1, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 1, 0, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 1, 0, 1, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 1, 1, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 0, 1, 1, 1, &vl, &vh, &elsum, &ehsum);
    // n==1 tau_new
    contribution_from(ed_cube, 1, 0, 0, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 0, 0, 1, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 0, 1, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 0, 1, 1, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 1, 0, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 1, 0, 1, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 1, 1, 0, &vl, &vh, &elsum, &ehsum);
    contribution_from(ed_cube, 1, 1, 1, 1, &vl, &vh, &elsum, &ehsum);

    if ( fabs(elsum) > acu ) vl /= elsum;
    if ( fabs(ehsum) > acu ) vh /= ehsum;
    return vl - vh;
}

// get the mass center for all the points
inline real4 get_mass_center(__private real4 * all_points, int num_points) {
    real4 center = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    for ( int i = 0; i < num_points; i++ ) {
        center += all_points[i];
    }
    return center/num_points;
}

// select half of the hyper surface with norm*energy_flow>0,
// calc the area for all the selected hypersf on the convex hull
// notice that no dx, dy, dz, dt, tau yet, dimensionless 
real4 calc_area(__private real4 *all_points, real4 energy_flow, 
                int num_points) {
    real4 mass_center = get_mass_center(all_points, num_points);
    real4 area = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
    hypersf sf;
    if ( num_points == 4 ) {
        sf = construct_hypersf(all_points[0], all_points[1],
                           all_points[2], all_points[3], mass_center);
        // if only 4 points, the norm vector has 2 possible directions
        if ( dot(energy_flow, sf.norm) < 0.0f ) sf.norm = -sf.norm;
        area = sf.norm;
    } else if ( num_points > 4 ) {
        // get all hyper surfaces, remove those not on convex hull
        for ( int i = 0; i < num_points-4; i ++ )
        for ( int j = i+1; j < num_points-3; j ++ )
        for ( int k = i+2; k < num_points-2; k ++ )
        for ( int l = i+3; l < num_points-1; l ++ ) {
            sf = construct_hypersf(all_points[i], all_points[j], all_points[k],
                                   all_points[l], mass_center);
             if ( is_on_convex_hull(sf, mass_center, all_points,
                                    num_points, i, j, k, l) ) {
                 area += sf.norm;
             }
        }
    }
    return area;
}

// Get the position of the intersection point on the edges of the cube
void ints_between(real ed_left, real ed_right, real4 pos_left,
                        real4 pos_right, __private real4 all_ints[32],
                        __private int num_points[1]) {
    real dE1 = EFRZ - ed_left;
    real dE2 = EFRZ - ed_right;
    real dE12 = ed_left - ed_right;
    if ( dE1*dE2 < 0 ) {
        real ratio = fabs(dE1)/fabs(dE12);
        all_ints[num_points[0]] = ratio*pos_right + (1-ratio)*pos_left;
        num_points[0] += 1;
    }
}

// Get all the intersections by comparing EFRZ with ed on the cube
void get_all_intersections(__private real ed[16],
               __private real4 all_ints[32],
               __private int num_points[1]) {
    num_points[0] = 0;
    // 16 edges with the same (z, tau)
    for (int start = 0; start < 16; start += 4) {
        ints_between(ed[start+0], ed[start+1], cube[start+0], cube[start+1],
                     all_ints, num_points);
        ints_between(ed[start+1], ed[start+2], cube[start+1], cube[start+2],
                     all_ints, num_points);
        ints_between(ed[start+2], ed[start+3], cube[start+2], cube[start+3],
                     all_ints, num_points);
        ints_between(ed[start+3], ed[start+0], cube[start+3], cube[start+0],
                     all_ints, num_points);
    }
    // 8 edges with the same (x, y, tau)
    for (int start = 0; start < 16; start += 8) {
        ints_between(ed[start+0], ed[start+4], cube[start+0], cube[start+4],
                     all_ints, num_points);
        ints_between(ed[start+1], ed[start+5], cube[start+1], cube[start+5],
                     all_ints, num_points);
        ints_between(ed[start+2], ed[start+6], cube[start+2], cube[start+6],
                     all_ints, num_points);
        ints_between(ed[start+3], ed[start+7], cube[start+3], cube[start+7],
                     all_ints, num_points);
    }
    // 8 edges with same (x, y, z) but different tau
    for ( int start = 0; start < 8; start ++) {
        ints_between(ed[start+0], ed[start+8], cube[start+0], cube[start+8],
                     all_ints, num_points);
    }
 }




// // return the index in the global array
// // I, J, K: thread id along x, y, z
// // i, j, k: 3d pos in d_ev[] array
// inline int idn(int I, int J, int K) {
//     int i = I*nx_skip;
//     int j = J*ny_skip;
//     int k = K*nz_skip;
//     return i*NY*NZ + j*NZ + k;
// }


// output: global d_hypersf array;
__kernel void test_hypersf(__global real4 * result) {
    __private real ed_cube[16];
    real4 mass_center;
    for (int i = 0; i < 8; i++) {
        ed_cube[i] = 2.0f;
        ed_cube[8+i] = 3.0f;
    }
    
    int num_of_intersection[1];
    
    __private real4 all_ints[32];

    get_all_intersections(ed_cube, all_ints, num_of_intersection);
    printf("num_of_intersection=%d\n", num_of_intersection[0]);
    
    real4 energy_flow_vector = energy_flow(ed_cube);

    mass_center = get_mass_center(all_ints, num_of_intersection[0]);

    real4 d_Sigma = calc_area(all_ints, energy_flow_vector, num_of_intersection[0]);

     result[0] = d_Sigma;
}
