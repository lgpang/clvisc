#include<real_type.h>

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
    (real4)(0.0f, 0.0f, 1.0f, 0.0f),
    (real4)(0.0f, 1.0f, 1.0f, 0.0f),
    
    (real4)(0.0f, 0.0f, 0.0f, 1.0f),
    (real4)(0.0f, 1.0f, 0.0f, 1.0f),
    (real4)(0.0f, 0.0f, 1.0f, 1.0f),
    (real4)(0.0f, 1.0f, 1.0f, 1.0f),
    // tau = tau_new, 8 corners cube
    (real4)(1.0f, 0.0f, 0.0f, 0.0f),
    (real4)(1.0f, 1.0f, 0.0f, 0.0f),
    (real4)(1.0f, 0.0f, 1.0f, 0.0f),
    (real4)(1.0f, 1.0f, 1.0f, 0.0f),
    
    (real4)(1.0f, 0.0f, 0.0f, 1.0f),
    (real4)(1.0f, 1.0f, 0.0f, 1.0f),
    (real4)(1.0f, 0.0f, 1.0f, 1.0f),
    (real4)(1.0f, 1.0f, 1.0f, 1.0f)
};

typedef struct {
    int   id[4];      // id for 4 points in the intersection list
    real4 center;     // surface center 
    real4 norm;       // outward norm vector
} hypersf;


// get the center of all the intersections
inline real4 get_mass_center(__private real4 * ints, int size_of_ints);

// calc the center and outward norm vector with four intersections
hypersf construct_hypersf(int id0, int id1, int id2, int id3,
                      __private real4 * ints, real4 mass_center);

real rand(int* seed);

// do tiny move if any of 5 intersections coplanar
void tiny_move_if_coplanar(__private real4 * ints, int size_of_ints,
                           real4 * mass_center, int gid);

// contribute of the energy density at cube corner to the energy flow vector
void contribution_from(__private real ed_cube[16], int n, int i, int j, int k,
                   real4 *vl, real4 *vh, real *elsum, real *ehsum);

// get the energy flow direction
real4 energy_flow(__private real ed_cube[16]);

// get the total area of all the hypersurface on the convex hull whose norm
// vector is in the same direction of energy flow
real4 calc_area(__private real4 *ints, real4 energy_flow,
                int size_of_ints, int gid);

void get_all_intersections(__private real ed[16],
               __private real4 all_ints[32],
               __private int size_of_ints[1]);

real centroid_pimn(__global real * d_pi_old, __global real* d_pi_new, real4 mc,
                   int mn, int i, int j, int k);
// get the ourward norm vector of one hyper surface
// mass_center is the center for all intersections
// vector_out = surf_center - mass_center
// norm_out * vector_out > 0; if possible for norm_out==0.0f
hypersf construct_hypersf(int id0, int id1, int id2, int id3,
                      __private real4 * ints, real4 mass_center) {
    hypersf surf;
    surf.id[0] = id0;
    surf.id[1] = id1;
    surf.id[2] = id2;
    surf.id[3] = id3;
    surf.center = 0.25f*(ints[id0] + ints[id1] + ints[id2] + ints[id3]);
    real4 vector_out = surf.center - mass_center;
    // the 3 vector that spans the hypersf
    real4 a = ints[id1] - ints[id0];
    real4 b = ints[id2] - ints[id0];
    real4 c = ints[id3] - ints[id0];
    // norm_vector has 2 directions
    real4 norm_vector = (real4)(
		 a.s1*(b.s2*c.s3-b.s3*c.s2) + a.s2*(b.s3*c.s1-b.s1*c.s3) + a.s3*(b.s1*c.s2-b.s2*c.s1),
        -(a.s0*(b.s2*c.s3-b.s3*c.s2) + a.s2*(b.s3*c.s0-b.s0*c.s3) + a.s3*(b.s0*c.s2-b.s2*c.s0)),
		 a.s0*(b.s1*c.s3-b.s3*c.s1) + a.s1*(b.s3*c.s0-b.s0*c.s3) + a.s3*(b.s0*c.s1-b.s1*c.s0),
		-(a.s0*(b.s1*c.s2-b.s2*c.s1) + a.s1*(b.s2*c.s0-b.s0*c.s2) + a.s2*(b.s0*c.s1-b.s1*c.s0)));
    
    real projection = dot(vector_out, norm_vector);
    if ( projection < 0 ) norm_vector = - norm_vector;
    surf.norm = norm_vector;       // surf.norm = dS_{\mu}
    return surf;
}

// very simple random number generator to provid some random tiny shift
// in case 5 points are coplanar.
real rand(int* seed) // 1 <= *seed < m
{
    int const a = 16807; //ie 7**5
    int const m = 2147483647; //ie 2**31-1
    *seed = ((long)(*seed * a))%m;
    return (*seed)/((real)m);
}


void tiny_move_if_coplanar(__private real4 * ints, int size_of_ints,
                           real4 * mass_center, int gid) {
   hypersf sf;
   // don't initialize seed with 0; otherwise seed will always be 0
   int seed = 32117*(gid+1);
   for ( int i = 0; i < size_of_ints-4; i ++ )
   for ( int j = i+1; j < size_of_ints-3; j ++ )
   for ( int k = j+1; k < size_of_ints-2; k ++ )
   for ( int l = k+1; l < size_of_ints-1; l ++ )
   for ( int m = l+1; m < size_of_ints; m ++ ){
       sf = construct_hypersf(i, j, k, l, ints, *mass_center);
       if ( dot(sf.norm, ints[m]-sf.center) == 0.0f ) {
          for ( int n = 0; n < size_of_ints; n ++ ) {
            real4 tiny_shift = (real4)((rand(&seed)-0.5f)*1.0E-7f,
                                  (rand(&seed)-0.5f)*1.0E-7f,
                                  (rand(&seed)-0.5f)*1.0E-7f,
                                  (rand(&seed)-0.5f)*1.0E-7f);
            ints[n] += tiny_shift;
          }

          * mass_center = get_mass_center(ints, size_of_ints);
          break;  // one random tiny move for all ints should work
       }
   }
}

// check if there are points beyond sf, if not return true
// n is the number of intersections
bool is_on_convex_hull(hypersf sf, real4 * mass_center,
                       __private real4 * ints,
                       int size_of_ints){
    for ( int n = 0; n < size_of_ints; n++ ) {
        if ( n != sf.id[0] && n != sf.id[1] && n != sf.id[2] && n != sf.id[3] ) {
            real4 test_vector = ints[n] - sf.center;
            // if there are points beyond sf, sf is not on convex
            if ( dot(test_vector, sf.norm) >= 0.0f ) {
                return false;
            }
        }
    }
    return true;
}

// get the weight of energy from each corner of the cube
// n, i, j, k can only be 0 or 1
void contribution_from(__private real ed_cube[16], int n, int i, int j, int k,
                              real4 *vl, real4 *vh, real *elsum, real *ehsum) {
    int id = 8*n + 4*k + 2*j + i;
    real dek = EFRZ - ed_cube[id];
    real adek = fabs(dek);
    if ( dek > 0.0f ) {
        *elsum += adek;
        *vl +=  (real4)(n, i, j, k)*adek;
    } else {
        *ehsum += adek;
        *vh +=  (real4)(n, i, j, k)*adek;
    }
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
inline real4 get_mass_center(__private real4 * ints, int size_of_ints) {
    real4 center = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    for ( int i = 0; i < size_of_ints; i++ ) {
        center += ints[i];
    }
    return center/size_of_ints;
}

// select half of the hyper surface with norm*energy_flow>0,
// calc the area for all the selected hypersf on the convex hull
// notice that no dx, dy, dz, dt, tau yet, dimensionless 
real4 calc_area(__private real4 *ints, real4 energy_flow, 
                int size_of_ints, int gid) {
    real4 mass_center = get_mass_center(ints, size_of_ints);
    real4 area = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
    hypersf sf;
    if ( size_of_ints == 4 ) {
        sf = construct_hypersf(0, 1, 2, 3, ints, mass_center);
        // if only 4 points, the norm vector has 2 possible directions
        if ( dot(energy_flow, sf.norm) < 0.0f ) sf.norm = -sf.norm;
        area = sf.norm;
    } else if ( size_of_ints > 4 ) {
        // do tiny move if any 4 points coplanar
        tiny_move_if_coplanar(ints, size_of_ints, & mass_center, gid);

        // get all hyper surfaces, remove those not on convex hull
        for ( int i = 0; i < size_of_ints-3; i ++ )
        for ( int j = i+1; j < size_of_ints-2; j ++ )
        for ( int k = j+1; k < size_of_ints-1; k ++ )
        for ( int l = k+1; l < size_of_ints; l ++ ) {
            sf = construct_hypersf(i, j, k, l, ints, mass_center);
            if ( is_on_convex_hull(sf, &mass_center, ints, size_of_ints)
                 && dot(sf.norm, energy_flow) > 0.0f  ) {
                 area += sf.norm;
                 // printf("used (i,j,k,l=%d,%d,%d,%d", i, j, k, l);
            }
        }
    }
    return area/6.0f;
}

// Get the position of the intersection point on the edges of the cube
void ints_between(real ed_left, real ed_right, real4 pos_left,
                        real4 pos_right, __private real4 all_ints[32],
                        __private int size_of_ints[1]) {
    real dE1 = EFRZ - ed_left;
    real dE2 = EFRZ - ed_right;
    real dE12 = ed_left - ed_right;
    if ( dE1*dE2 < 0 ) {
        real ratio = fabs(dE1)/fabs(dE12);
        all_ints[size_of_ints[0]] = ratio*pos_right + (1-ratio)*pos_left;
        size_of_ints[0] += 1;
    }
}

// Get all the intersections by comparing EFRZ with ed on the cube
void get_all_intersections(__private real ed[16],
               __private real4 all_ints[32],
               __private int size_of_ints[1]) {
    size_of_ints[0] = 0;
    // 16 edges with the same (z, tau)
    for (int start = 0; start < 16; start += 4) {
        ints_between(ed[start+0], ed[start+1], cube[start+0], cube[start+1],
                     all_ints, size_of_ints);
        ints_between(ed[start+0], ed[start+2], cube[start+0], cube[start+2],
                     all_ints, size_of_ints);
        ints_between(ed[start+2], ed[start+3], cube[start+2], cube[start+3],
                     all_ints, size_of_ints);
        ints_between(ed[start+1], ed[start+3], cube[start+1], cube[start+3],
                     all_ints, size_of_ints);
    }
    // 8 edges with the same (x, y, tau)
    for (int start = 0; start < 16; start += 8) {
        ints_between(ed[start+0], ed[start+4], cube[start+0], cube[start+4],
                     all_ints, size_of_ints);
        ints_between(ed[start+1], ed[start+5], cube[start+1], cube[start+5],
                     all_ints, size_of_ints);
        ints_between(ed[start+2], ed[start+6], cube[start+2], cube[start+6],
                     all_ints, size_of_ints);
        ints_between(ed[start+3], ed[start+7], cube[start+3], cube[start+7],
                     all_ints, size_of_ints);
    }
    // 8 edges with same (x, y, z) but different tau
    for ( int start = 0; start < 8; start ++) {
        ints_between(ed[start+0], ed[start+8], cube[start+0], cube[start+8],
                     all_ints, size_of_ints);
    }
 }


// return the index in the global array
// I, J, K: thread id along x, y, z
// i, j, k: 3d pos in d_ev[] array
inline int idn(int I, int J, int K) {
    int i = I*nxskip;
    int j = J*nyskip;
    int k = K*nzskip;
    return i*NY*NZ + j*NZ + k;
}

// get the ed, v at the mass center by 4d interpolation
// mc means mass_center
real4 centroid_ev(__private real4 ev_cube[16], real4 mc){
    real4 centroid;
    centroid = (1.0f - mc.s0)*(1.0f - mc.s1)*(1.0f - mc.s2)*(1.0f - mc.s3)*ev_cube[0]
             + (1.0f - mc.s0)*mc.s1*(1.0f - mc.s2)*(1.0f - mc.s3)*ev_cube[1]
             + (1.0f - mc.s0)*mc.s1*mc.s2*(1.0f - mc.s3)*ev_cube[2]
             + (1.0f - mc.s0)*(1.0f - mc.s1)*mc.s2*(1.0f - mc.s3)*ev_cube[3]
             + (1.0f - mc.s0)*(1.0f - mc.s1)*(1.0f - mc.s2)*mc.s3*ev_cube[4]
             + (1.0f - mc.s0)*mc.s1*(1.0f - mc.s2)*mc.s3*ev_cube[5]
             + (1.0f - mc.s0)*mc.s1*mc.s2*mc.s3*ev_cube[6]
             + (1.0f - mc.s0)*(1.0f - mc.s1)*mc.s2*mc.s3*ev_cube[7]
             + mc.s0*(1.0f - mc.s1)*(1.0f - mc.s2)*(1.0f - mc.s3)*ev_cube[8]
             + mc.s0*mc.s1*(1.0f - mc.s2)*(1.0f - mc.s3)*ev_cube[9]
             + mc.s0*mc.s1*mc.s2*(1.0f - mc.s3)*ev_cube[10]
             + mc.s0*(1.0f - mc.s1)*mc.s2*(1.0f - mc.s3)*ev_cube[11]
             + mc.s0*(1.0f - mc.s1)*(1.0f - mc.s2)*mc.s3*ev_cube[12]
             + mc.s0*mc.s1*(1.0f - mc.s2)*mc.s3*ev_cube[13]
             + mc.s0*mc.s1*mc.s2*mc.s3*ev_cube[14]
             + mc.s0*(1.0f - mc.s1)*mc.s2*mc.s3*ev_cube[15];
    return centroid;
}


real centroid_pimn(__global real * d_pi_old, __global real* d_pi_new, real4 mc,
                   int mn, int i, int j, int k){
    real centroid;
    centroid = (1.0f - mc.s0)*(1.0f - mc.s1)*(1.0f - mc.s2)*(1.0f - mc.s3)*d_pi_old[idn(i, j, k)*10+mn]
             + (1.0f - mc.s0)*mc.s1*(1.0f - mc.s2)*(1.0f - mc.s3)*d_pi_old[idn(i+1, j, k)*10+mn]
             + (1.0f - mc.s0)*mc.s1*mc.s2*(1.0f - mc.s3)*d_pi_old[idn(i, j+1, k)*10+mn]
             + (1.0f - mc.s0)*(1.0f - mc.s1)*mc.s2*(1.0f - mc.s3)*d_pi_old[idn(i+1, j+1, k)*10+mn]
             + (1.0f - mc.s0)*(1.0f - mc.s1)*(1.0f - mc.s2)*mc.s3*d_pi_old[idn(i, j, k+1)*10+mn]
             + (1.0f - mc.s0)*mc.s1*(1.0f - mc.s2)*mc.s3*d_pi_old[idn(i+1, j, k+1)*10+mn]
             + (1.0f - mc.s0)*mc.s1*mc.s2*mc.s3*d_pi_old[idn(i, j+1, k+1)*10+mn]
             + (1.0f - mc.s0)*(1.0f - mc.s1)*mc.s2*mc.s3*d_pi_old[idn(i+1, j+1, k+1)*10+mn]
             + mc.s0*(1.0f - mc.s1)*(1.0f - mc.s2)*(1.0f - mc.s3)*d_pi_new[idn(i, j, k)*10+mn]
             + mc.s0*mc.s1*(1.0f - mc.s2)*(1.0f - mc.s3)*d_pi_new[idn(i+1, j, k)*10+mn]
             + mc.s0*mc.s1*mc.s2*(1.0f - mc.s3)*d_pi_new[idn(i, j+1, k)*10+mn]
             + mc.s0*(1.0f - mc.s1)*mc.s2*(1.0f - mc.s3)*d_pi_new[idn(i+1, j+1, k)*10+mn]
             + mc.s0*(1.0f - mc.s1)*(1.0f - mc.s2)*mc.s3*d_pi_new[idn(i, j, k+1)*10+mn]
             + mc.s0*mc.s1*(1.0f - mc.s2)*mc.s3*d_pi_new[idn(i+1, j, k+1)*10+mn]
             + mc.s0*mc.s1*mc.s2*mc.s3*d_pi_new[idn(i, j+1, k+1)*10+mn]
             + mc.s0*(1.0f - mc.s1)*mc.s2*mc.s3*d_pi_new[idn(i+1, j+1, k+1)*10+mn];
    return centroid;
}


// calc the hypersf and store them in one buffer DA0, DA1, DA2, DA3,
// vx, vy, veta, tau, x, y, eta
__kernel void get_hypersf(__global real8  * d_sf,
                          __global int * num_of_sf,
                          __global real4 * d_ev_old,
                          __global real4 * d_ev_new,
                          const real time_old,
                          const real time_new) {
    __private real4 ev_cube[16];
    __private real ed_cube[16];
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    real4 mass_center;
    int num_of_intersection;
    real4 d_Sigma;

    bool is_surf = true;
    if ( (i+1)*nxskip < NX && (j+1)*nyskip < NY && (k+1)*nzskip < NZ ) {
        ev_cube[0] = d_ev_old[idn(i, j, k)];
        ev_cube[1] = d_ev_old[idn(i+1, j, k)];
        ev_cube[2] = d_ev_old[idn(i, j+1, k)];
        ev_cube[3] = d_ev_old[idn(i+1, j+1, k)];

        ev_cube[4] = d_ev_old[idn(i, j, k+1)];
        ev_cube[5] = d_ev_old[idn(i+1, j, k+1)];
        ev_cube[6] = d_ev_old[idn(i, j+1, k+1)];
        ev_cube[7] = d_ev_old[idn(i+1, j+1, k+1)];

        ev_cube[8] = d_ev_new[idn(i, j, k)];
        ev_cube[9] = d_ev_new[idn(i+1, j, k)];
        ev_cube[10] = d_ev_new[idn(i, j+1, k)];
        ev_cube[11] = d_ev_new[idn(i+1, j+1, k)];

        ev_cube[12] = d_ev_new[idn(i, j, k+1)];
        ev_cube[13] = d_ev_new[idn(i+1, j, k+1)];
        ev_cube[14] = d_ev_new[idn(i, j+1, k+1)];
        ev_cube[15] = d_ev_new[idn(i+1, j+1, k+1)];

        for ( int grid = 0; grid < 16; grid ++ ) {
            ed_cube[grid] = ev_cube[grid].s0;
        }

       
        if ( (ed_cube[0] - EFRZ)*(ed_cube[15] - EFRZ) > 0 && 
             (ed_cube[1] - EFRZ)*(ed_cube[14] - EFRZ) > 0 &&
             (ed_cube[2] - EFRZ)*(ed_cube[13] - EFRZ) > 0 &&
             (ed_cube[3] - EFRZ)*(ed_cube[12] - EFRZ) > 0 &&
             (ed_cube[4] - EFRZ)*(ed_cube[11] - EFRZ) > 0 &&
             (ed_cube[5] - EFRZ)*(ed_cube[10] - EFRZ) > 0 &&
             (ed_cube[6] - EFRZ)*(ed_cube[9] - EFRZ) > 0 &&
             (ed_cube[7] - EFRZ)*(ed_cube[8] - EFRZ) > 0 ) {
            is_surf = false;
        }
    
        if ( is_surf ) {
            __private real4 all_ints[32];

            get_all_intersections(ed_cube, all_ints, &num_of_intersection);
            real4 energy_flow_vector = energy_flow(ed_cube);
            mass_center = get_mass_center(all_ints, num_of_intersection);
            d_Sigma = calc_area(all_ints, energy_flow_vector,
                                num_of_intersection, i*j*k);

            real dtd = time_new - time_old;
            real dxd = nxskip*DX;
            real dyd = nyskip*DY;
            real dzd = nzskip*DZ;

            // space-time centroid of the freeze out hyper-surface
            real tau = time_old + mass_center.s0*dtd;
            real x = -NX/2*DX + i*dxd + mass_center.s1*dxd;
            real y = -NY/2*DY + j*dyd + mass_center.s2*dyd;
            real eta = -NZ/2*DZ + k*dzd + mass_center.s3*dzd;

            real4 ev = centroid_ev(ev_cube, mass_center);

            real8 result = (real8)(tau*dxd*dyd*dzd*d_Sigma.s0,
                                  -tau*dtd*dyd*dzd*d_Sigma.s1,
                                  -tau*dtd*dxd*dzd*d_Sigma.s2,
                                  -dtd*dxd*dyd*d_Sigma.s3,
                                   ev.s1, ev.s2, ev.s3, eta);
                                   
            d_sf[atomic_inc(num_of_sf)] = result;
        } // end surface calculation
    } // end boundary check
}

// calc the hypersf for viscous hydro and store them in one buffer 
//DA0, DA1, DA2, DA3, vx, vy, veta, tau, x, y, eta
//and another one for pimn terms
__kernel void visc_hypersf(__global real8  * d_sf,
                           __global real   * d_pi,
                          __global int * num_of_sf,
                          __global real4 * d_ev_old,
                          __global real4 * d_ev_new,
                          __global real * d_pi_old,
                          __global real * d_pi_new,
                          const real time_old,
                          const real time_new) {
    __private real4 ev_cube[16];
    __private real ed_cube[16];
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    real4 mass_center;
    int num_of_intersection;
    real4 d_Sigma;

    bool is_surf = true;
    if ( (i+1)*nxskip < NX && (j+1)*nyskip < NY && (k+1)*nzskip < NZ ) {
        ev_cube[0] = d_ev_old[idn(i, j, k)];
        ev_cube[1] = d_ev_old[idn(i+1, j, k)];
        ev_cube[2] = d_ev_old[idn(i, j+1, k)];
        ev_cube[3] = d_ev_old[idn(i+1, j+1, k)];

        ev_cube[4] = d_ev_old[idn(i, j, k+1)];
        ev_cube[5] = d_ev_old[idn(i+1, j, k+1)];
        ev_cube[6] = d_ev_old[idn(i, j+1, k+1)];
        ev_cube[7] = d_ev_old[idn(i+1, j+1, k+1)];

        ev_cube[8] = d_ev_new[idn(i, j, k)];
        ev_cube[9] = d_ev_new[idn(i+1, j, k)];
        ev_cube[10] = d_ev_new[idn(i, j+1, k)];
        ev_cube[11] = d_ev_new[idn(i+1, j+1, k)];

        ev_cube[12] = d_ev_new[idn(i, j, k+1)];
        ev_cube[13] = d_ev_new[idn(i+1, j, k+1)];
        ev_cube[14] = d_ev_new[idn(i, j+1, k+1)];
        ev_cube[15] = d_ev_new[idn(i+1, j+1, k+1)];

        for ( int grid = 0; grid < 16; grid ++ ) {
            ed_cube[grid] = ev_cube[grid].s0;
        }

       
        if ( (ed_cube[0] - EFRZ)*(ed_cube[15] - EFRZ) > 0 && 
             (ed_cube[1] - EFRZ)*(ed_cube[14] - EFRZ) > 0 &&
             (ed_cube[2] - EFRZ)*(ed_cube[13] - EFRZ) > 0 &&
             (ed_cube[3] - EFRZ)*(ed_cube[12] - EFRZ) > 0 &&
             (ed_cube[4] - EFRZ)*(ed_cube[11] - EFRZ) > 0 &&
             (ed_cube[5] - EFRZ)*(ed_cube[10] - EFRZ) > 0 &&
             (ed_cube[6] - EFRZ)*(ed_cube[9] - EFRZ) > 0 &&
             (ed_cube[7] - EFRZ)*(ed_cube[8] - EFRZ) > 0 ) {
            is_surf = false;
        }
    
        if ( is_surf ) {
            __private real4 all_ints[32];

            get_all_intersections(ed_cube, all_ints, &num_of_intersection);
            real4 energy_flow_vector = energy_flow(ed_cube);
            mass_center = get_mass_center(all_ints, num_of_intersection);
            d_Sigma = calc_area(all_ints, energy_flow_vector,
                                num_of_intersection, i*j*k);

            real dtd = time_new - time_old;
            real dxd = nxskip*DX;
            real dyd = nyskip*DY;
            real dzd = nzskip*DZ;

            // space-time centroid of the freeze out hyper-surface
            real tau = time_old + mass_center.s0*dtd;
            real x = -NX/2*DX + i*dxd + mass_center.s1*dxd;
            real y = -NY/2*DY + j*dyd + mass_center.s2*dyd;
            real eta = -NZ/2*DZ + k*dzd + mass_center.s3*dzd;

            real4 ev = centroid_ev(ev_cube, mass_center);

            real8 result = (real8)(tau*dxd*dyd*dzd*d_Sigma.s0,
                                  -tau*dtd*dyd*dzd*d_Sigma.s1,
                                  -tau*dtd*dxd*dzd*d_Sigma.s2,
                                  -dtd*dxd*dyd*d_Sigma.s3,
                                   ev.s1, ev.s2, ev.s3, eta);

            int id_ = atomic_inc(num_of_sf);
                                   
            d_sf[id_] = result;
            d_pi[10*id_ + 0] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 0, i, j, k);
            d_pi[10*id_ + 1] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 1, i, j, k);
            d_pi[10*id_ + 2] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 2, i, j, k);
            d_pi[10*id_ + 3] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 3, i, j, k);
            d_pi[10*id_ + 4] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 4, i, j, k);
            d_pi[10*id_ + 5] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 5, i, j, k);
            d_pi[10*id_ + 6] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 6, i, j, k);
            d_pi[10*id_ + 7] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 7, i, j, k);
            d_pi[10*id_ + 8] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 8, i, j, k);
            d_pi[10*id_ + 9] = centroid_pimn(d_pi_old, d_pi_new, mass_center, 9, i, j, k);
        } // end surface calculation
    } // end boundary check
}



// unit test for hypersf calc with a test cube
// output: global d_hypersf array;
__kernel void test_hypersf(__global real4 * result) {
    __private real ed_cube[16];
    for (int i = 0; i < 8; i++) {
        ed_cube[i] = 3.0f;
        ed_cube[8+i] = 2.0f;
    }
    
    int num_of_intersection;
    
    __private real4 all_ints[32];

    get_all_intersections(ed_cube, all_ints, &num_of_intersection);

    //printf("num_of_intersections=%d\n", num_of_intersection);
    
    real4 energy_flow_vector = energy_flow(ed_cube);

    // real4 mass_center;
    // mass_center = get_mass_center(all_ints, num_of_intersection);

    int gid = get_global_id(0);
    real4 d_Sigma = calc_area(all_ints, energy_flow_vector,
                              num_of_intersection, gid);

    result[0] = d_Sigma;
}



