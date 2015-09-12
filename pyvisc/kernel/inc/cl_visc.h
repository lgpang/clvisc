#ifndef __CL_MATRIX__
#define __CL_MATRIX__

#define __CL_ENABLE_EXCEPTIONS /*!< Use cpp exceptionn to handel errors */
// System includes
#include <CL/cl.hpp>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <ctime>
#include <algorithm>

#include <random>

#include <Config.h>
#include <CEos.h>

//#include "cl.hpp"

#ifdef USE_SINGLE_PRECISION
 typedef cl_float cl_real;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float4 cl_real4;
 typedef cl_float3 cl_real3;
 typedef cl_float16 cl_real16;
#else
 typedef cl_double cl_real;
 typedef cl_double4 cl_real4;
 typedef cl_double3 cl_real3;
 typedef cl_double16 cl_real16;
#endif


/*! \class Visc 
 *  \breif Viscous hydro in opencl gpu parallel
 *  */
class Visc
{
    private:
	cl::Context context;
	std::vector<cl::Device> devices;
	std::vector<cl::Program> programs;
	cl::CommandQueue queue;
	cl::Kernel kernel_initIdeal;
	cl::Kernel kernel_stepIdeal;
	cl::Kernel kernel_globIdeal;

	cl::Kernel kernel_initVisc;
	cl::Kernel kernel_stepVisc;
	cl::Kernel kernel_stepVisc1;
	cl::Kernel kernel_src2;
	cl::Kernel kernel_globVisc;
	cl::Kernel kernel_regulatePimn;

	cl::Kernel kernel_bulk;  
	cl::Kernel kernel_max;  /*! get the maximum energy density */
	cl_uint2  nseed;       /*! random seed used in CPU side */
	cl_uint   Size;    /**< Size = Nx*Ny*Nz */

    //// Buffer for ideal part
	cl::Buffer d_Hm0, d_Tm0, d_Umu1, d_Ed, d_Pr, d_Src;
    cl::Buffer d_NewTm00, d_NewUmu, d_NewEd;

    /** \breif Buffer for visc hydro. 
     * half step: d_pi0 from n -> n+1/2, use d_Ed(at n), d_Umu0(at n-1/2), d_Umu1( at n )
     * full step: d_pi1 from n -> n+1  , use d_Ed(at n+1/2), d_Umu0( at n),d_Umu1( at n+1/2)
     * */
    cl::Buffer d_Umu0, d_pi0, d_pi1, d_Newpi;
    cl::Buffer d_Sigma; // Sigma^{mu nu} for tt, xx, yy, hh

    /** \breif Eos table (ed, P, T, S) for interpolation */
    cl::Buffer d_EosTable;

	/*! \breif helper functions: create context from the device type with one platform which support it 
	 * \return one context in the type of cl::Context
	 */
    cl::Context CreateContext( const cl_int & device_type );   

	/*! \breif helper functions: Read *.cl source file and append it to programs vector 
	 */
	void AddProgram( const char * fname );

	/*! \breif helper functions: Build each program in programs vector with given options
	 *  \note The compiling error of the kernel *.cl can be print out
	 */
	void BuildPrograms( const char * compile_options );

    void runKernelVisc( const cl_real & tau, const int & half_step, const cl::NDRange & globalSize, const cl::NDRange & localSize );

    void runKernelIdeal( const cl_real & tau, const int & half_step , const cl::NDRange & globalSize, const cl::NDRange & localSize );

    void updateGlobMem( const cl_real & tau, const int & half_step , const cl::NDRange & globalSize, const cl::NDRange & localSize );


    public:
	std::vector<cl_real4> h_Umu0; 
	std::vector<cl_real4> h_Umu1; 
	std::vector<cl_real> h_Ed;   
	std::vector<cl_real> h_Pr;   

	std::vector<cl_real> h_pi0;   
	std::vector<cl_real> h_pi1;   

	std::stringstream str_buff; /*! buff for all the parton information at each time step */


	Visc();

	~Visc();

	/*! \breif Set hydro parameters */
	void ReadConfigs( const int & ieos, const int & nx, const int &ny, const int & nz, const int & ntskip, const int & nxskip, const int & nyskip, const int & nzskip, const cl_real & dt, const cl_real & dx, const cl_real & dy, const cl_real & dz, const cl_real & tau0, const cl_real & TFRZ, const cl_real & etaos, const cl_real & lam1H, const std::string & outputdir );

    int NX, NY, NZ, NTSKIP, NXSKIP, NYSKIP, NZSKIP;
    cl_real DT, DX, DY, DZ;
    cl_real ETAOS, TAU0, LAM1H;
    std::string fPathOut;

	/*! \breif Read Ed, Umu from inited.dat */
	void ReadIniCond( const char * fInicond0, const char * fInicond1 );        

    /** \breif EosTable has data ( ed, P, T, S ) on host **/
    cl_int IEOS;  /** IEOS==0 for p=e/3; ==4 for EOSL_CE; ==5 for EOSL_PCE; */
    CEos * Eos;
	std::vector<cl_real4> h_EosTable;   
    /*! \breif Set Eos */
    void SetEos( const int & ieos );

    cl_real TFRZ;
    void SetTfrz( const cl_real & TFrz0 );

	/*! \breif CreateContext, AddProgram, Build program, Initialize Buffer*/
	void initializeCL();    


    void runKernelInit();

	/*! \breif Use GPU device to run hydro evolution with KT algorithm*/
	void runKernelEvolve();

    cl_real getEdMax();

    void output( const cl_real & tau, const int & n, const int & ntskip, const int & nxskip, const int & nyskip, const int & nzskip );

    void bulkInfo( const cl_real & tau, const int & n, const int & nskip );

	void testResults();           /*!< Test function */

	void clean(); /*!< delete pointers if there is any */

};


#endif 


/*! \mainpage 
 *  \section  */

/*! 
 *  \example 
 *  This example tells you how to use colljet. Actually 
 *  most of the jobs are done by subroutines in class Colljet,
 *  you can change initial jet information by modify SetIniJet.
 *  and Change bulk information by "void SetBulkInfo( cl_real time);".
 *  This is the first version.
 *
 *  \code
 *
 *


 *   \endcode
*/


