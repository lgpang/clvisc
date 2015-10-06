#ifndef __CL_MATRIX__
#define __CL_MATRIX__

#define __CL_ENABLE_EXCEPTIONS /*!< Use cpp exceptionn to handel errors */
// System includes
//#include <CL/cl.hpp>
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
#include <map>

#include <random>

#include <Config.h>
#include "cl.hpp"

#ifdef USE_SINGLE_PRECISION
 typedef cl_float cl_real;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float2 cl_real2;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float4 cl_real4;
 typedef cl_float3 cl_real3;
 typedef cl_float8 cl_real8;
#else
 typedef cl_double cl_real;
 typedef cl_double2 cl_real2;
 typedef cl_double4 cl_real4;
 typedef cl_double3 cl_real3;
 typedef cl_double8 cl_real8;
#endif

#include "Int.h"

#define NY 41
#define NPT 15 
#define NPHI 48  
#define YLO -8.0
#define YHI 8.0
#define INVSLOPE 1.0/12.0


typedef struct {
    cl_int  pidR;
    cl_real branch;
    cl_int  numpart;
    cl_real part[5];
}CDecay;

struct CParticle{
    int monval;
    std::string  name;
    cl_real mass;
    cl_real width; 
    cl_real	gspin;      /* spin degeneracy */
    cl_int	baryon;     /* baryon number   */
    cl_int	strange;
    cl_int	charm;
    cl_int	bottom;
    cl_int	gisospin;  /* isospin degeneracy */
    cl_int	charge;    
    cl_int	decays;    /* amount of decays listed for this resonance */
    cl_int	stable;     /* defines whether this particle is considered as stable */
    //int Num;              /*How many particles are produced */
    //real4 * pmu;         /* Four momentum */
    //real4 * xmu;         /* space time infor */
    /// If antibaryon_spec_exists == 1, just copy, don't repeat calculation
    cl_int  antibaryon_spec_exists;
};



/*! \class Spec 
 *  \breif Specous hydro in opencl gpu parallel
 *  */
class Spec
{
    private:
	cl::Context context;
	std::vector<cl::Device> devices;
	std::vector<cl::Program> programs;
	cl::CommandQueue queue;
	cl::Kernel kernel_subspec;
	cl::Kernel kernel_spec;

	cl::Kernel kernel_decay2;
	cl::Kernel kernel_decay3;
	cl::Kernel kernel_sumdecay;

	cl_uint   SizeSF;    /**< Size = h_SF.size() */
	cl_uint   SizePID;    /**< Size = h_SF.size() */

	cl::Buffer d_SF;
    cl::Buffer d_SubSpec;
    cl::Buffer d_Spec;
    cl::Buffer d_HadronInfo;
    cl::Buffer d_Y;
    cl::Buffer d_Pt;
    cl::Buffer d_CPhi;
    cl::Buffer d_SPhi;

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
	//void BuildPrograms( const char * compile_options );
	void BuildPrograms( int program_id, const char * compile_options );

	/*! \breif CreateContext, AddProgram, Build program, Initialize Buffer*/
	void initializeCL();    

    
    void AddReso(cl_int pidR, cl_int j, cl_int k, \
        std::vector<cl_real> &branch, std::vector<cl_real4> &mass , \
        std::vector<cl_int>  &resoNum,std::vector<cl_real > &norm3);

    void getDecayInfo( cl_int pid, cl_int nbody, \
            std::vector<cl_real> &branch, std::vector<cl_real4> &mass , \
            std::vector<cl_int>  &resoNum,std::vector<cl_real > &norm3);

    void get_dNdEtaPtdPtdPhi(cl_real mass, cl_real dNdYPtdPtdPhi[NY][NPT][NPHI], \
            cl_real dNdEtaPtdPtdPhi[NY][NPT][NPHI]);

    public:

    cl_real Tfrz;
    ///chemical potential from PCE-EOS
    std::map<cl_int, cl_real> muB;

    std::map< cl_int, cl_int > newpid ;   /** map monval to position in the array */
	std::vector<cl_real8> h_SF; 
	std::vector<cl_real> h_SubSpec;       /** spec for all particles divided into 13 groups */
	std::vector<cl_real> h_Spec;          /** do summation for the subspec; */
	std::vector<cl_real4> h_HadronInfo;    /**< s[0]-> mass, s[1]-> gspin */

	std::vector<CDecay> decay;    /**< decay info read from pdg */

	std::vector<cl_real> h_Y;                /** constant memory: rapidity */ 
	std::vector<cl_real> h_Pt;               /** constant memory: pt */
	std::vector<cl_real> h_CPhi;              /** constant memory: phi */
	std::vector<cl_real> h_SPhi;              /** constant memory: phi */

    std::vector<CParticle> particles;

	std::stringstream str_buff; /*! buff for all the parton information at each time step */

	Spec();

	~Spec();

	void ReadMuB(  const std::string & muB_datafile );        

	void ReadHyperSF(  const std::string & sf_datafile );        

    void ReadParticles( char * particle_data_table);

    std::string  DataPath;

    void SetPathOut( const std::string & path );

    /*! \breif Set freeze out temperature */
    void SetTfrz( const cl_real & Tfrz );

    void InitGrid(const cl_int & Nrapidity, const cl_real& Ylo, const cl_real & Yhi);

    void CalcSpec(bool switch_off_decay);   /*!< Calc dNdYPtdPtdPhi for each particle */

    void ReadSpec();              /*!< Read dNdYPtdPtdPhi for each particle */

    void ResoDecay();             /*!< perform resonance decay on GPU */

	void testResults();           /*!< Test function */

	void clean(); /*!< delete pointers if there is any */

};


#endif 


/*! \mainpage 
 *  \section  */

/*! 
 *  \example 
*
 *  \code
 *
 *


 *   \endcode
*/


