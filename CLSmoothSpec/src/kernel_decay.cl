#include<helper.h>
#define PTCHANGE ((real)1.0)

#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define ACU ((real)1.0E-12f)

#define PTN1 20
#define PTN2 20

//{{{ quadruature integral nodes
__constant    double    gaulep8[] = {   0.9602898564,   0.7966664774, 0.5255324099, 0.1834346424    };
__constant    double    gaulew8[] = {   0.1012285362,   0.2223810344, 0.3137066458, 0.3626837833    };
__constant	double	gaulep20[] = {
		0.993128599185094,	0.963971927277913,
		0.912234428251325,	0.839116971822218,
		0.746331906460150,	0.636053680726515,
		0.510867001950827,	0.373706088715419,
		0.227785851141645,	0.076526521133497	};
__constant double gaulew20[] = {
		0.017614007139152,	0.040601429800386,
		0.062672048334109,	0.083276741576704,
		0.101930119817240,	0.118194531961518,
		0.131688638449176,	0.142096109318382,
		0.149172986472603,	0.152753387130725	};

__constant   double p12[] = {	0.9815606342,	0.9041172563, 0.7699026741,	0.5873179542, 0.3678314989,	0.1252334085	};
__constant   double w12[] = {	0.0471753363,	0.1069393259, 0.1600783285,	0.2031674267, 0.2334925365,	0.2491470458	};
/** Gauss Quadrature integral nodes and weights */
__constant	double gaulep48[] = {
    0.998771007252426118601,	0.993530172266350757548,
    0.984124583722826857745,	0.970591592546247250461,
    0.952987703160430860723,	0.931386690706554333114,
    0.905879136715569672822,	0.876572020274247885906,
    0.843588261624393530711,	0.807066204029442627083,
    0.767159032515740339254,	0.724034130923814654674,
    0.677872379632663905212,	0.628867396776513623995,
    0.577224726083972703818,	0.523160974722233033678,
    0.466902904750958404545,	0.408686481990716729916,
    0.348755886292160738160,	0.287362487355455576736,
    0.224763790394689061225,	0.161222356068891718056,
    0.097004699209462698930,	0.032380170962869362033 };

__constant	double	gaulew48[] = {
    0.003153346052305838633,	0.007327553901276262102,
    0.011477234579234539490,	0.015579315722943848728,
    0.019616160457355527814,	0.023570760839324379141,
    0.027426509708356948200,	0.031167227832798088902,
    0.034777222564770438893,	0.038241351065830706317,
    0.041545082943464749214,	0.044674560856694280419,
    0.047616658492490474826,	0.050359035553854474958,
    0.052890189485193667096,	0.055199503699984162868,
    0.057277292100403215705,	0.059114839698395635746,
    0.060704439165893880053,	0.062039423159892663904,
    0.063114192286254025657,	0.063924238584648186624,
    0.064466164435950082207,	0.064737696812683922503 };

__constant 	double gala15x[] = {	0.093307812017,         0.492691740302,
    1.215595412071,         2.269949526204,
    3.667622721751,         5.425336627414,
    7.565916226613,        10.120228568019, 
    13.130282482176,        16.654407708330,
    20.776478899449,        25.623894226729,
    31.407519169754,        38.530683306486,
    48.026085572686	};
__constant	double gala15w[] = {	0.239578170311,         0.560100842793,
    0.887008262919,         1.22366440215,
    1.57444872163,          1.94475197653,
    2.34150205664,          2.77404192683,
    3.25564334640,          3.80631171423,
    4.45847775384,          5.27001778443,
    6.35956346973,          8.03178763212,
    11.5277721009   };
//}}}

/** 1D linear interpolation */
inline real lin_int( real x1, real x2, real y1, real y2, real x )
{
    real r = ( x - x1 )/ (x2-x1 );
    return y1*(1.0f-r) + y2*r ;
}


/** interpolation function to get spec at arbitrary Y,Pt and Phi */
double Edndp3( real * yr, real * ptr, real * phir, int pidR, __global real * d_Spec )
{
    double val = 1.0E-25;
    real y0, y1, pt0, pt1, phi0, phi1;

    real dY = (YHI- YLO)/(real)(NY-1);

    int offset = (pidR) * NY*NPT*NPHI;


    
    if ( (*yr>=YLO) && (*yr<=YHI) && (*ptr<8.0f) ){

        int nry = 1;
        while(true){
            y0 = YLO + (nry-1)*dY ;
            y1 = YLO + nry * dY ;
            if( (*yr<y1) || (nry==NY-1) ) break;
            nry ++;
        }

        int npt = 1;
        while( true ){
            pt0 = INVSLOPE*gala15x[npt-1] ;
            pt1 = INVSLOPE*gala15x[npt] ;
            if( (*ptr<pt1) || (npt==NPT-1) ) break;
            npt ++;
        }

        int nphi = 1;
        while( true ) {
            phi0 = ((nphi-1)<NPHI/2) ? M_PI_F*(1.0-gaulep48[nphi-1]) : M_PI_F*(1.0+gaulep48[NPHI-1-(nphi-1)]);
            phi1 = (nphi<NPHI/2) ? M_PI_F*(1.0-gaulep48[nphi]) : M_PI_F*(1.0+gaulep48[NPHI-1-nphi]);
            if( (*phir < phi1) || (nphi==NPHI-1) ) break;
            nphi ++;
        }

        double fy1, fy2;
        double f1, f2;


        fy1 = lin_int(phi0, phi1,  
                d_Spec[offset + (nry-1)*NPT*NPHI + (npt-1)*NPHI + (nphi-1)], 
                d_Spec[offset + (nry-1)*NPT*NPHI + (npt-1)*NPHI + (nphi)  ], *phir);
        fy2 = lin_int(phi0, phi1,
                d_Spec[offset + (nry)*NPT*NPHI + (npt-1)*NPHI + (nphi-1)], 
                d_Spec[offset + (nry)*NPT*NPHI + (npt-1)*NPHI + (nphi)  ], *phir);

        f1 = lin_int(y0, y1, fy1, fy2, *yr);

        fy1 = lin_int(phi0, phi1,
                d_Spec[offset + (nry-1)*NPT*NPHI + (npt)*NPHI + (nphi-1)], 
                d_Spec[offset + (nry-1)*NPT*NPHI + (npt)*NPHI + (nphi)  ], *phir);
        fy2 = lin_int(phi0, phi1,
                d_Spec[offset + (nry)*NPT*NPHI + (npt)*NPHI + (nphi-1)], 
                d_Spec[offset + (nry)*NPT*NPHI + (npt)*NPHI + (nphi)  ], *phir);

        f2 = lin_int(y0, y1, fy1, fy2, *yr);

        //printf( "yr, ptr, phir=%d, %d, %d\n", nry, npt, nphi );

        int use_log = 0;
        if(*ptr > PTCHANGE){
            f1 = (f1>2.0E-307)?f1:2.0E-307;
            f2 = (f2>2.0E-307)?f2:2.0E-307;
            f1 = log(f1); 
            f2 = log(f2);
            use_log = 1;
        }
        /* pt interpolation */
        val = lin_int(pt0, pt1, f1, f2, *ptr);

        //if(*ptr > PTCHANGE)  val = exp(val);
        if ( use_log == 1 ){
             if ( val > 10.0 ) val = 10.0;
             val = exp(val);
        }

        val = (val>2.0E-307)?val:2.0E-307;

    } else{
        val = 0.0f;
    }

    return  val;

}


typedef struct pblockN
{
    real pt, mt, y, e, pl;  /* pt, mt, y of decay product 1 */
    real phi;
    real m1, m2, m3;        /* masses of decay products     */
    real mr;            /* mass of resonance            */
    real costh, sinth;    /* integrate over y */
    real e0, p0;          /* E* , p* of decay productu 1 in local rest frame of resonance */
    int res_num;            /* Montecarlo number of the Res. */
} pblockN;


//real Edndp3( real yr, real ptr, real phir,  int res_num, __global real * d_Spec );


//The Main integrand for integration. 
real dnpir2N (real phi, real costh, real w2, real y, real pt, real phi1, real m1, real m2, real mr, int reso_num, __global real * d_Spec )
{

    real mt = sqrt( pt*pt + m1 * m1 );
    real e  = mt * cosh( y );
    real pl = mt * sinh( y );

    //real w2 = m2*m2;

    real e0 = (mr * mr + m1 * m1 - w2) / (2 * mr);
    real p0 = sqrt(max(ACU, e0 * e0 - m1 * m1));

    real sinth = sqrt(max(ACU, (real)1.0 - costh * costh));

    real D;
    real eR, plR, ptR, yR, phiR, sume, jac;
    real cphiR, sphiR;


    sume = e + e0;

    D = max(ACU, e * e0 + pl * p0 * costh + pt * p0 * sinth * cos (phi) + m1 * m1);

    eR = mr * (sume * sume/D - 1.0);
    jac = mr + eR;
    plR = mr * sume * (pl - p0 * costh) /D;
    ptR = (eR * eR - plR * plR - mr * mr);

    if (ptR < 1.0E-15f) ptR = 1.0E-15f;
    else ptR = sqrt (ptR);

    //yR = 0.5 * log (max(ACU, eR + plR)/max(ACU, (eR - plR)));
    yR = 0.5 * (log(max(ACU, eR + plR))-log(max(ACU, eR - plR)));

    /// possible non from /(sume*ptR)
    cphiR = -jac * (p0 * sinth * cos (phi + phi1) - pt * cos (phi1)) / (sume * ptR);
    sphiR = -jac * (p0 * sinth * sin (phi + phi1) - pt * sin (phi1)) / (sume * ptR);

    if ((fabs (cphiR) > 1.0f))
    {
        if (cphiR > 0.01f)
            cphiR = 0.99999f;
        if (cphiR < -0.01f)
            cphiR = -0.99999f;
    }

    phiR = atan2(sphiR, cphiR);

    phiR = (phiR >= 0) ? phiR : (2.0f*M_PI_F+phiR);

    double dnr = Edndp3( &yR, &ptR, &phiR, reso_num, d_Spec);

    // jac = mr * sume*sume /D
    //real result = dnr * jac * jac / (2.0 * sume * sume) ;
    //real result = dnr * min(1.0E30, mr*mr*sume*sume/(D*D) * 0.5);

    double jac_coef = mr*mr*sume*sume/(D*D)*0.5f;


    if ( isfinite(jac_coef) && !isnan(jac_coef) ) jac_coef = min(1000.0, jac_coef);
    else jac_coef = 0.0;

    //if ( jac_coef > 1000.0 && dnr > 0.1 ) {
    //   printf( "yr, ptr, phir=%f, %f, %f\n", yR, ptR, phiR );
    //   printf( "dnr = %f\n", dnr );
    //   printf( "jac = %f\n", jac_coef);
    //   printf( "D = %f\n", D );
    //   printf( "cosPhiR = %f\n", cphiR );
    //   printf( "pidR = %d\n", reso_num );
    //}

    real result = isfinite(dnr) ? dnr*jac_coef : 0.0;

    if ( !isfinite(result) || isnan(result) || ptR > 10.0 ) {
       result = 0.0;
    }

    if (result < 0.0 ) result = 0.0;

    return result;
}


/** Integrate over phiR */
real dnpir1N (real costh, real w2, real y, real pt, real phi, real m1, real m2, real mr, int reso_num, __global real * d_Spec )
{

    real xlo = 0.0;
    real xhi = 2.0*M_PI_F;

    real xoffs = 0.5 * ( xlo + xhi );
    real xdiff = 0.5 * ( xhi - xlo );
    real s = 0.0;

    if (PTN2 == 20) {
        for( int ix=0; ix<10; ix ++ )
            s += gaulew20[ ix ] * ( dnpir2N( xoffs + xdiff*gaulep20[ix], costh, w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) + \
                    dnpir2N( xoffs - xdiff*gaulep20[ix], costh, w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) );
    
    } else if (PTN2 == 8) {
        for( int ix=0; ix<4; ix ++ )
            s += gaulew8[ ix ] * ( dnpir2N( xoffs + xdiff*gaulep8[ix], costh, w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) + \
                    dnpir2N( xoffs - xdiff*gaulep8[ix], costh, w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) );
    }

    return s * xdiff;
}


/** \bug  void * may be needed to change to other things 
  integrate over costhR from -1 to 1 */
real dn2ptN ( real w2, real y, real pt, real phi, real m1, real m2, real mr, int reso_num, __global real * d_Spec )
{
    //pblockN *para = (pblockN *) para1;

    real xlo = -1.0f;
    real xhi =  1.0f;

    real xoffs = 0.5f * ( xlo + xhi );
    real xdiff = 0.5f * ( xhi - xlo );
    real s = 0.0;

//    #pragma unroll 4
    if (PTN1 == 20) {
        for( int ix=0; ix<10; ix ++ ) {
             s += gaulew20[ix] * (dnpir1N( xoffs + xdiff*gaulep20[ix], w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) + \
                    dnpir1N( xoffs - xdiff*gaulep20[ix], w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ));
        }
    } else if (PTN1 == 8) {
        for( int ix=0; ix<4; ix ++ )
            s += gaulew8[ ix ] * ( dnpir1N( xoffs + xdiff*gaulep8[ix], w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) + \
                    dnpir1N( xoffs - xdiff*gaulep8[ix], w2, y, pt, phi, m1, m2, mr, reso_num, d_Spec ) );
    }

    return s * xdiff;
}



real Edndp3_2bodyN( real y, real pt, real phi, real m1, real m2, real mr, int reso_num,  __global real * d_Spec )
{
    real norm2 = 1.0 / (2.0*M_PI_F);
    return norm2 * dn2ptN( m2*m2, y, pt, phi, m1, m2, mr, reso_num, d_Spec );
}


/** First stage: calc sub spec for each particle */
__kernel void decay_2body(  
        __global real  * d_Spec,            
        __global real  * d_Decay,
        __global real4 * d_Mass,           
        __global real  * d_BranchRatio,           
        __global int   * d_ResNum,           
        const int pid
        )
{
    int nchannels = get_num_groups(0);
    int i, j, k, pidR;
    real y, pt, phi;
    real4 mass;

    int  n = get_local_id(0);

    while( n < NY*NPT*NPHI ){
        i = n / (NPT * NPHI );
        j = (n-i*NPT*NPHI) / NPHI ;
        k = n-i*NPT*NPHI - j*NPHI ;
        pidR = d_ResNum[ get_group_id(0) ];
        mass = d_Mass[ get_group_id(0) ];

        y = YLO + i * (YHI-YLO)/(real)(NY-1) ;
        pt = INVSLOPE * gala15x[ j ];
        phi = (k<NPHI/2) ? M_PI_F*(1.0-gaulep48[k]) : M_PI_F*(1.0+gaulep48[NPHI-1-k]);


        real mr = mass.s0;
        real m1 = mass.s1;
        real m2 = mass.s2;

        d_Decay[ n * nchannels + get_group_id(0) ] =  d_BranchRatio[get_group_id(0)] * Edndp3_2bodyN( y, pt, phi, m1, m2, mr, pidR, d_Spec );

        n += get_local_size(0);
    }

}
////////////////End of 2 body decay ////////////////////////////////

real dn3ptN( real x, real y, real pt, real phi, real m1, real m2, real m3, real mr, int reso_num, __global real * d_Spec )
{
/*here x = W**2, should be bigger than 0.0 */

real e0 = ( mr*mr + m1*m1 - x ) / ( 2.0*mr );
real p0 = sqrt(max(ACU, e0*e0 - m1*m1));

real a = ( m2 + m3 ) * ( m2 + m3 );
real b = ( m2 - m3 ) * ( m2 - m3 );

/** no check on x */
return p0 * sqrt(max(ACU, (x-a)*(x-b)))/max(x, ACU) * dn2ptN( x, y, pt, phi, m1, m2, mr, reso_num, d_Spec );

}

real Edndp3_3bodyN( real y, real pt, real phi, real m1, real m2, real m3, real mr, real norm3, int reso_num,  __global real * d_Spec )
{

    real xlo = ( m2 + m3 ) * ( m2 + m3 );
    real xhi = ( mr - m1 ) * ( mr - m1 );

    real xoffs = 0.5 * ( xlo + xhi );
    real xdiff = 0.5 * ( xhi - xlo );
    real s = 0.0;


/** \breif use unroll to expand for loop */
    for( int ix=0; ix<10; ix ++ ){
        s += gaulew20[ ix ] * ( dn3ptN(xoffs + xdiff*gaulep20[ix], y, pt, phi, m1, m2, m3, mr, reso_num, d_Spec) + \
                dn3ptN(xoffs - xdiff*gaulep20[ix],  y, pt, phi, m1, m2, m3, mr, reso_num, d_Spec));
    }
    real res3 = 2.0 * norm3 * (s * xdiff) / mr;
    if (res3 < 0.0f) res3 = 0.0f;
    return res3;
}



__kernel void decay_3body(  
        __global real  * d_Spec,            
        __global real  * d_Decay,
        __global real4 * d_Mass,           
        __global real  * d_BranchRatio,           
        __global real  * d_norm3,           
        __global int   * d_ResNum,           
        const int pid)
{
    int nchannels = get_num_groups(0);
    int i, j, k;
    real y, pt, phi;

    int  n = get_local_id(0);
    int  pidR = d_ResNum[ get_group_id(0) ];
    real4 mass = d_Mass[ get_group_id(0) ];
    real norm3 = d_norm3[ get_group_id(0) ];
    real mr = mass.s0;
    real m1 = mass.s1;
    real m2 = mass.s2;
    real m3 = mass.s3;

    while( n < NY*NPT*NPHI ){
        i = n / (NPT * NPHI );
        j = (n-i*NPT*NPHI) / NPHI ;
        k = n-i*NPT*NPHI - j*NPHI ;

        y = YLO + i * (YHI-YLO)/(real)(NY-1) ;
        pt = INVSLOPE * gala15x[ j ];
        phi = (k<NPHI/2) ? M_PI_F*(1.0-gaulep48[k]) : M_PI_F*(1.0+gaulep48[NPHI-1-k]);

        d_Decay[ n * nchannels + get_group_id(0) ] =  d_BranchRatio[get_group_id(0)] * Edndp3_3bodyN( y, pt, phi, m1, m2, m3, mr, norm3, pidR, d_Spec );

        n += get_local_size(0);
    }

}

/*! \breif Kham summation to reduce accumate error */
inline real khamSum(__global real *data, int offset, int size)
{
    real sum = data[offset];
    real c = 0.0f;
    for (int i = 1; i < size; i++)
    {
        real y = data[offset + i] - c;
        real t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    if( isnan(sum) ) sum = 0.0;
    return sum;
}


/** The second stage on sum reduction 
 *   For each Y,Pt,Phi we need to sum nchannels dNdYptdptdphi from the first stage */
__kernel void sumResoDecay(  
        __global real  * d_Decay,            
        __global real  * d_Spec,
        const int nchannels,
        const int pid)
{
    int I = get_global_id(0);
    /** I in range [0 ,  NY*NPT*NPHI) */

    d_Spec[ pid*NY*NPT*NPHI + I ] += khamSum( d_Decay, I*nchannels, nchannels );

}
