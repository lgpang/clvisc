#include <cl_visc.h>

//#define DT 0.01
//#define DX 0.04
//#define DY 0.04
//#define DZ 0.3
//#define NX 201 
//#define NY 201
//#define NZ 6
//#define ETAOS 0.2

//#define LAM1H 1.0
#define VISCOUSON

#define DEVICE_ID 0

/*! \breif Kahan summation to reduce accumate error */
  template<class T>
T reduceCPU(T *data, int size)
{
  T sum = data[0];
  T c = (T)0.0;
  for (int i = 1; i < size; i++)
  {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

//////////// Set initial random seed for each threads ///////////////
int ReadRandomSeed(){
  std::ifstream frand("/dev/urandom", std::ios::in | std::ios::binary );
  int m;
  frand.read((char*)(&m),sizeof(m));
  return m;
}


Visc::Visc()
{
  TFRZ = 0.137;  /** Default value; Can be changed by SetTfrz( Tfrz ) */
}


Visc::~Visc()
{
}

cl::Context Visc::CreateContext( const cl_int & device_type )
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get( &platforms );
  if( platforms.size() == 0 ){
    std::cerr<<"no platform found\n";
    std::exit(-1);
  }
  else{
    for( int i=0; i < platforms.size(); i++ ){
      std::vector<cl::Device> supportDevices;
      platforms.at(i).getDevices( CL_DEVICE_TYPE_ALL, & supportDevices );
      for(int j=0; j < supportDevices.size(); j++ ){
        if( supportDevices.at(j).getInfo<CL_DEVICE_TYPE>() == device_type ){
          std::cout<<"#Found device "<<device_type<<" on platform "<<i<<std::endl;
          cl_context_properties properties[] =
          { CL_CONTEXT_PLATFORM, 
            (cl_context_properties) (platforms.at(i))(),
            0 };
          return cl::Context( device_type, properties );
        }// Found supported device and platform
      }// End for devices
    }// End for platform
    //// if no platform support device_type, exit
    std::cerr<<"no platform support device type"<<device_type<<std::endl;
    std::exit( -1 );
  }
}

void Visc::AddProgram( const char * fname)
{ //// An compact way to add program from file
  std::ifstream kernelFile( fname );
  if( !kernelFile.is_open() ) std::cerr<<"Open "<<fname << " failed!"<<std::endl;

  std::string sprog( std::istreambuf_iterator<char> (kernelFile), (std::istreambuf_iterator<char> ()) );
  cl::Program::Sources prog(1, std::make_pair(sprog.c_str(), sprog.length()));

  programs.push_back( cl::Program( context, prog ) );

}

void Visc::BuildPrograms( const char * compile_options )
{ //// build programs and output the compile error if there is
  std::string build_log;
  for(std::vector<cl::Program>::size_type i=0; i!=programs.size(); i++)
  {
    try{
      programs.at(i).build(devices, compile_options);
    }
    catch(cl::Error & err){
      std::cout<< programs.at(i).getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cout<<err.what()<<"("<<err.err()<<")\n";
    }
  }

}

/*! \breif Set hydro parameters */
void Visc::ReadConfigs( const int & ieos, const int & nx, const int &ny, const int & nz, const int & ntskip, const int & nxskip, const int & nyskip, const int & nzskip, const cl_real & dt, const cl_real & dx, const cl_real & dy, const cl_real & dz, const cl_real & tau0, const cl_real & Tfrz, const cl_real & etaos , const cl_real & lam1H, const std::string & outputdir)
{
  IEOS = ieos;
  NX = nx; NY = ny; NZ = nz;
  NTSKIP = ntskip;
  NXSKIP = nxskip;
  NYSKIP = nyskip;
  NZSKIP = nzskip;
  DT = dt; DX = dx; DY = dy; DZ = dz;
  TAU0 = tau0;
  TFRZ = Tfrz;
  ETAOS = etaos;
  LAM1H = lam1H;
  fPathOut = outputdir;
}


///////////// Set Equation of state ////////////////////////////////////////////
void Visc::SetEos( const int & ieos ){
  IEOS = ieos;
  if( ieos == 0 ) Eos = new CEosI();
  else if( ieos == 4 ) Eos = new CEosL( "CE" );
  else if( ieos == 5 ) Eos = new CEosL("PCE" );
  else std::cerr<<"Eos "<<ieos<<" not implemented yet\n";

  cl_real eps;
  cl_real4 EPTS;
  for( int i=0; i<2000; i++ ){
    eps = i * 0.03;
    EPTS = (cl_real4) { eps, Eos->P(eps,0.0),  Eos->T(eps,0.0), Eos->S(eps,0.0) };
    h_EosTable.push_back( EPTS );
  }
}

void Visc::SetTfrz( const cl_real & Tfrz ){
  TFRZ = Tfrz;
}

/*! Set the information of the initial jet parton */
void Visc::ReadIniCond( const char * fInicond0, const char * fInicond1 )
{
  try {
    Size = NX*NY*NZ;
    cl_real ed, temp; 
    cl_real heta = 1.0;
    std::ifstream fed( fInicond0 );
    if( ! fed.is_open() ){
      std::cerr<<"Can not open fInicond0\n";
    }

    for(int i=0; i<NX; i++)
      for(int j=0; j<NY; j++){
        for(int k=0; k<NZ; k++){
          fed>>ed;
          h_Ed.push_back( ed );
          cl_real4 umu= (cl_real4){1.0, 0.0, 0.0, 0.0};
          h_Umu1.push_back( umu );
          h_Umu0.push_back( umu );

          for( int m=0; m<10; m++ ){
            h_pi1.push_back(0.0);
            h_pi0.push_back(0.0);
          }
        }
      }



    //    cl_real heta = 1.0;
    //    char buff[256];
    //
    //    std::ifstream fed0( fInicond0 );
    //    std::ifstream fed1( fInicond1 );
    //    if( ! fed0.is_open() ){
    //      std::cerr<<"Can not open fInicond0\n";
    //    }
    //    fed0.getline( buff, 256 );
    //    fed1.getline( buff, 256 );
    //
    //    cl_real tau, x, y, etas, ut, ux, uy;
    //    cl_real pitt, pitx, pity, pixx, pixy, piyy, pizz;
    //
    //
    //
    //    for(int i=0; i<NX; i++)
    //    for(int j=0; j<NY; j++)
    //    for(int k=0; k<NZ; k++){
    //        cl_real x = (i-NX/2) * DX;
    //        cl_real y = (j-NY/2) * DY;
    //        cl_real z = (k-NZ/2) * DZ;
    //        cl_real Ed = 30.0f * exp( - (x*x+y*y)/4.0 ) ;
    //        //cl_real Ed = 30.0f ;
    //        h_Ed.push_back( Ed );
    //        cl_real4 umu= (cl_real4){1.0, 0.0, 0.0, 0.0};
    //        h_Umu0.push_back( umu );
    //        h_Umu1.push_back( umu );
    //        for( int m=0; m<10; m++ ){
    //          h_pi1.push_back(0.0);
    //          h_pi0.push_back(0.0);
    //        }
    //    }

    /** Gubser initial condition */     
    //        cl_real eps0 = 1.0;
    //        cl_real tau = 1.0;
    //        cl_real q = 1.0;
    //
    //        for(int i=0; i<NX; i++)
    //        for(int j=0; j<NY; j++)
    //        for(int k=0; k<NZ; k++){
    //            cl_real x = (i-NX/2) * DX;
    //            cl_real y = (j-NY/2) * DY;
    //            cl_real z = (k-NZ/2) * DZ;
    //            cl_real rT = sqrt( x*x + y*y );
    //            cl_real Ed = eps0 * pow( 2.0*q, 8.0/3.0 )*pow(tau, -4.0/3.0) * pow( 1 + 2*q*q*(tau*tau+rT*rT) 
    //                    + pow( q, 4.0 ) * pow( tau*tau - rT*rT , 2.0), -4.0/3.0);
    //            h_Ed.push_back( Ed );
    //
    //            cl_real k = atanh( 2.0*q*q*tau*rT / ( 1.0 + q*q*tau*tau + q*q*rT*rT ) );
    //
    //            cl_real4 umu= (cl_real4){1.0, 0.0, 0.0, 0.0};
    //
    //            cl_real phi = atan2( y, x );
    //
    //            umu.s[0] = cosh( k );
    //            umu.s[1] = cos( phi ) * sinh( k );
    //            umu.s[2] = sin( phi ) * sinh( k );
    //            umu.s[3] = 0.0;
    //
    //            h_Umu.push_back( umu );
    //        }


    /**Gubser viscous initial condition */
    //      std::ifstream fin( "../gubser/iniEd.dat" );
    //      cl_real T, ut, ux, uy, ut0, ux0, uy0;
    //
    //      if( fin.is_open() ){
    //          for(int i=0; i<NX; i++){
    //              for(int j=0; j<NY; j++){
    //                  fin>>T>>ut0>>ux0>>uy0>>ut>>ux>>uy;
    //                  for(int k=0; k<NZ; k++){
    //                      h_Ed.push_back( Eos->Ed( T, 0.0 ) );
    //                      h_Umu0.push_back( (cl_real4) {ut0, ux0, uy0, 0.0} );
    //                      h_Umu1.push_back( (cl_real4) {ut, ux , uy , 0.0} );
    //                  }
    //              }
    //          }
    //      }
    //      else{
    //          std::cerr<<"Can not open iniEd.dat for read\n";
    //          exit(0);
    //      }


  }//end try
  catch (cl::Error &err ){
    std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
  }

}


void Visc::initializeCL()
{

  try {
    cl_int device_type = CL_DEVICE_TYPE_CPU;

#ifdef USE_DEVICE_GPU
    device_type = CL_DEVICE_TYPE_GPU;
#endif

    context = CreateContext( device_type );

    devices = context.getInfo<CL_CONTEXT_DEVICES>();

    for( std::vector<cl::Device>::size_type i=0; i!=devices.size(); i++){
      std::cout<<"#"<<devices[i].getInfo<CL_DEVICE_NAME>()<<'\n';
      std::cout<<"#Max compute units ="<<devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<'\n';
    }

    // Define compile-time constants, compute domains and types.
    std::stringstream compile_options;
    std::string dev_vendor = devices[DEVICE_ID].getInfo<CL_DEVICE_VENDOR>();
    std::cout<<"#dev_vendor="<<dev_vendor<<std::endl;

    compile_options << "-I../src"<<" ";
    //compile_options << "-cl-nv-verbose"<<" ";
    //compile_options << "-cl-fast-relaxed-math -DMAC"<<" ";
    //compile_options << "-cl-opt-disable"<<" ";

    if ( sizeof(cl_real) == 4 ){
      compile_options << "-D USE_SINGLE_PRECISION"<<" ";
    }

    if( IEOS == 0 ) compile_options << "-D EOSI"<<" ";   //for eos selection
    else if ( IEOS == 4 ) compile_options << "-D EOSLCE"<<" ";   //for eos selection
    else if ( IEOS == 5 ) compile_options << "-D EOSLPCE"<<" ";   //for eos selection
    compile_options << "-D BSZ="<<BSZ<<" ";  //local block size with 4 halo cells
    compile_options << "-D VSZ="<<BSZ-2<<" ";  //local block size with 2 halo cells
    compile_options << "-D DT=((real)"<<DT<<") ";  
    compile_options << "-D DX=((real)"<<DX<<") ";  
    compile_options << "-D DY=((real)"<<DY<<") ";  
    compile_options << "-D DZ=((real)"<<DZ<<") ";  
    compile_options << "-D NX="<<NX<<" ";  
    compile_options << "-D NY="<<NY<<" ";  
    compile_options << "-D NZ="<<NZ<<" ";  
    compile_options << "-D ETAOS=((real)"<<ETAOS<<") ";  
    compile_options << "-D LAM1H=((real)"<<LAM1H<<") ";  

    queue = cl::CommandQueue( context, devices[ DEVICE_ID ], CL_QUEUE_PROFILING_ENABLE );

    //std::cout<<"Vendor="<<dev_vendor<<'\n';

    AddProgram( "../src/kernel_ideal.cl" );

    AddProgram( "../src/kernel_bulk.cl" );

    AddProgram( "../src/kernel_reduction.cl" );

#ifdef VISCOUSON
    AddProgram( "../src/kernel_visc.cl" );
    AddProgram( "../src/kernel_src2.cl" );
#endif




    BuildPrograms( compile_options.str().c_str() );

    kernel_initIdeal = cl::Kernel( programs.at(0), "initIdeal" );
    kernel_stepIdeal = cl::Kernel( programs.at(0), "stepUpdate" );
    kernel_globIdeal = cl::Kernel( programs.at(0), "updateGlobalMem" );

    kernel_bulk = cl::Kernel( programs.at(1), "getBulkInfo" );
    kernel_max = cl::Kernel( programs.at(2), "reduction_stage1" );
    std::cout<<"#ideal compile program succeed\n";

#ifdef VISCOUSON
    kernel_initVisc = cl::Kernel( programs.at(3), "initVisc" );
    kernel_stepVisc1 = cl::Kernel( programs.at(3), "stepUpdateVisc1" );
    kernel_stepVisc = cl::Kernel( programs.at(3), "stepUpdateVisc" );
    kernel_globVisc = cl::Kernel( programs.at(3), "updateGlobalMemVisc" );
    kernel_regulatePimn = cl::Kernel( programs.at(3), "regulatePimn" );

    std::cout<<"#viscous compile program succeed\n";
    kernel_src2 = cl::Kernel( programs.at(4), "updateSrcFromPimn" );
#endif

    std::cout<<"#viscous src update comiple succeed\n";

    h_Pr.resize( Size );

    /** buffers for ideal part */
    d_Hm0 = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real4)); //global
    d_Tm0 = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real4)); //global
    d_Umu1 = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Size*sizeof(cl_real4), h_Umu1.data()); //global
    d_Src = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real4)); //global
    d_Ed = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Size*sizeof(cl_real), h_Ed.data()); //global

    d_NewTm00 = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real4)); //global
    d_NewUmu = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real4)); //global
    d_NewEd = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real)); //global
    d_Pr = cl::Buffer( context, CL_MEM_READ_WRITE, Size*sizeof(cl_real)); //global

    /** buffers for viscous part */

#ifdef VISCOUSON        
    d_Umu0 = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Size*sizeof(cl_real4), h_Umu0.data()); //global
    d_Sigma = cl::Buffer( context, CL_MEM_READ_WRITE , Size*sizeof(cl_real4)); //global
    d_pi0 = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10*Size*sizeof(cl_real), h_pi0.data()); //global
    d_pi1 = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10*Size*sizeof(cl_real), h_pi1.data()); //global
    d_Newpi = cl::Buffer( context, CL_MEM_READ_WRITE, 10*Size*sizeof(cl_real)); //global
#endif

    d_EosTable = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_EosTable.size()*sizeof(cl_real4), h_EosTable.data()); //global
  }
  catch (cl::Error & err ){
    std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
  }

}


//** \breif get the maximum energy density */
cl_real Visc::getEdMax()
{
  cl_real h_semimax[64];
  cl::Buffer d_semimax = cl::Buffer( context, CL_MEM_READ_WRITE, 64*sizeof( cl_real ) );

  int NGrids = h_Ed.size();
  try{
    kernel_max.setArg( 0, d_Ed );
    kernel_max.setArg( 1, d_semimax );
    kernel_max.setArg( 2, NGrids );
    cl::Event event;
    queue.enqueueNDRangeKernel( kernel_max, cl::NullRange, cl::NDRange(64*256), cl::NDRange(256), NULL, &event );
    event.wait();
    queue.enqueueReadBuffer( d_semimax, CL_TRUE, 0, 64*sizeof(cl_real), h_semimax );
  }
  catch(cl::Error &err){          
    std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
  }

  return * std::max_element( h_semimax, h_semimax+64 );


}

void Visc::runKernelInit()
{
  try{

    cl::NDRange globalSize = cl::NDRange( Size );
    cl::NDRange localSize = cl::NullRange;

    kernel_initIdeal.setArg( 0, d_Hm0 );
    kernel_initIdeal.setArg( 1, d_Tm0 );
    kernel_initIdeal.setArg( 2, d_Umu1 );
    kernel_initIdeal.setArg( 3, d_Src );
    kernel_initIdeal.setArg( 4, d_Ed  );
    kernel_initIdeal.setArg( 5, TAU0  );
    kernel_initIdeal.setArg( 6, Size  );
    cl::Event event;
    queue.enqueueNDRangeKernel( kernel_initIdeal, cl::NullRange, globalSize, localSize, NULL, &event);
    event.wait();
    std::cout<<"#Init ideal Finished "<<std::endl;

#ifdef VISCOUSON        
    cl::Event event1;
    kernel_initVisc.setArg( 0, d_pi0 );
    kernel_initVisc.setArg( 1, d_pi1 );
    kernel_initVisc.setArg( 2, d_Ed );
    kernel_initVisc.setArg( 3, d_Umu0 );
    kernel_initVisc.setArg( 4, d_Umu1 );
    kernel_initVisc.setArg( 5, TAU0 );
    kernel_initVisc.setArg( 6, Size );

    int LSZ = BSZ - 4;
    cl::NDRange globalSize1 = cl::NDRange( NX, NY, NZ );
    cl::NDRange  localSize1 = cl::NDRange( LSZ, LSZ, LSZ );

    //queue.enqueueNDRangeKernel( kernel_initVisc, cl::NullRange, globalSize, localSize, NULL, &event1);
    queue.enqueueNDRangeKernel( kernel_initVisc, cl::NullRange, globalSize1, localSize1, NULL, &event1);
    event1.wait();
    std::cout<<"#Init Visc finished\n";
#endif

  }                               
  catch(cl::Error &err){          
    std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
  }
}


void Visc::runKernelVisc( const cl_real & tau, const int & half_step, const cl::NDRange & globalSize, const cl::NDRange & localSize )
{

#ifdef VISCOUSON
  /** half step for pimn and update d_Src from 
   *
   * \f$ \partial_{mu} ( U^{mu} pi^{mu nu} ) = 0 \f$ */
  cl_real time = tau + DT;
  if( half_step ){
    kernel_stepVisc1.setArg( 0, d_pi0 );
    kernel_stepVisc1.setArg( 1, d_pi1 );
    kernel_stepVisc1.setArg( 2, d_Umu0 );
    kernel_stepVisc1.setArg( 3, d_Umu1 );
    kernel_stepVisc1.setArg( 4, d_Ed );
    kernel_stepVisc1.setArg( 5, d_Newpi );
    kernel_stepVisc1.setArg( 6, d_Src );
    kernel_stepVisc1.setArg( 7, d_Sigma );
    kernel_stepVisc1.setArg( 8, tau );
    kernel_stepVisc1.setArg( 9, half_step );
    kernel_stepVisc1.setArg( 10, Size );
    // d_pi1 is updated


    kernel_stepVisc.setArg( 0, d_pi1 );
    kernel_stepVisc.setArg( 1, d_pi1 );
    kernel_stepVisc.setArg( 2, d_Umu0 );
    kernel_stepVisc.setArg( 3, d_Umu1 );
    kernel_stepVisc.setArg( 4, d_Ed );
    kernel_stepVisc.setArg( 5, d_Newpi );
    kernel_stepVisc.setArg( 6, d_Src );
    kernel_stepVisc.setArg( 7, tau );
    kernel_stepVisc.setArg( 8, half_step );
    kernel_stepVisc.setArg( 9, Size );
    // d_Newpi is updated

    //kernel_src2.setArg( 0, d_pi1 );
    kernel_src2.setArg( 0, d_Newpi );
    kernel_src2.setArg( 1, d_Umu1 );
    kernel_src2.setArg( 3, tau );   
  }
  else{
    kernel_stepVisc1.setArg( 0, d_pi1 );
    kernel_stepVisc1.setArg( 1, d_Newpi );
    kernel_stepVisc1.setArg( 2, d_Umu1 );
    kernel_stepVisc1.setArg( 3, d_NewUmu );
    kernel_stepVisc1.setArg( 4, d_NewEd );
    kernel_stepVisc1.setArg( 5, d_pi1 );
    kernel_stepVisc1.setArg( 6, d_Src );
    kernel_stepVisc1.setArg( 7, d_Sigma );
    kernel_stepVisc1.setArg( 8, tau );
    kernel_stepVisc1.setArg( 9, half_step );
    kernel_stepVisc1.setArg( 10, Size );
    //d_Newpi is updated

    kernel_stepVisc.setArg( 0, d_Newpi );
    kernel_stepVisc.setArg( 1, d_Newpi );
    //kernel_stepVisc.setArg( 1, d_pi0 );
    kernel_stepVisc.setArg( 2, d_Umu1 );
    kernel_stepVisc.setArg( 3, d_NewUmu );
    kernel_stepVisc.setArg( 4, d_NewEd );
    kernel_stepVisc.setArg( 5, d_pi1 );
    kernel_stepVisc.setArg( 6, d_Src );
    kernel_stepVisc.setArg( 7, tau );
    kernel_stepVisc.setArg( 8, half_step );
    kernel_stepVisc.setArg( 9, Size );
    //d_pi1 is updated

    //kernel_src2.setArg( 0, d_Newpi );
    kernel_src2.setArg( 0, d_pi1 );
    kernel_src2.setArg( 1, d_NewUmu );
    kernel_src2.setArg( 3, time );   
  }

  cl::Event event_stepVisc1;
  queue.enqueueNDRangeKernel( kernel_stepVisc1, cl::NullRange, globalSize, localSize, NULL, &event_stepVisc1); 
  event_stepVisc1.wait();

  std::vector<cl_real4> h_Sigma;
  h_Sigma.resize( Size );
  queue.enqueueReadBuffer( d_Sigma, CL_TRUE, 0, Size*sizeof(cl_real4), h_Sigma.data() );

  cl_real4 sigmamm = h_Sigma.at(Size/2);
  std::cout<<"sigma_xx="<<sigmamm.s[1]<<" sigma_zz"<<sigmamm.s[3]<<std::endl;



  cl::Event event_stepVisc;
  queue.enqueueNDRangeKernel( kernel_stepVisc, cl::NullRange, globalSize, localSize, NULL, &event_stepVisc); 
  event_stepVisc.wait();

  /** Calc src from isolated pi^{mn} in \partial_{mu} T^{mu nu} */
  /* \note It seems move kernel_src before kernel_stepVisc1 will make the program stable 
   * but the energy density evolution will be not correct */
  cl::Event event_src2;
  kernel_src2.setArg( 2, d_Src );
  kernel_src2.setArg( 4, Size );
  queue.enqueueNDRangeKernel( kernel_src2, cl::NullRange, globalSize, localSize, NULL, &event_src2);
  event_src2.wait();


#endif

}


void Visc::runKernelIdeal( const cl_real & tau, const int & half_step , const cl::NDRange & globalSize, const cl::NDRange & localSize )
{
  if( half_step ){
    kernel_stepIdeal.setArg( 0, d_Hm0 );
    kernel_stepIdeal.setArg( 1, d_Tm0 );
    kernel_stepIdeal.setArg( 2, d_Umu1 );
    kernel_stepIdeal.setArg( 3, d_Src );
    kernel_stepIdeal.setArg( 4, d_Ed  );
    kernel_stepIdeal.setArg( 5, d_NewTm00  );
    kernel_stepIdeal.setArg( 6, d_NewUmu  );
    kernel_stepIdeal.setArg( 7, d_NewEd  );
    kernel_stepIdeal.setArg( 8, tau  );
    kernel_stepIdeal.setArg( 9, half_step     );
    kernel_stepIdeal.setArg( 10, Size  );
  }
  else{
    kernel_stepIdeal.setArg( 0, d_NewTm00 );
    kernel_stepIdeal.setArg( 1, d_Hm0 );
    kernel_stepIdeal.setArg( 2, d_NewUmu );
    kernel_stepIdeal.setArg( 3, d_Src );
    kernel_stepIdeal.setArg( 4, d_NewEd  );
    kernel_stepIdeal.setArg( 5, d_Tm0 );
    kernel_stepIdeal.setArg( 6, d_Umu1  );
    kernel_stepIdeal.setArg( 7, d_Ed  );
    kernel_stepIdeal.setArg( 8, tau  );
    kernel_stepIdeal.setArg( 9, half_step     );
    kernel_stepIdeal.setArg( 10, Size  );
  }

  cl::Event event_halfstep;
  queue.enqueueNDRangeKernel( kernel_stepIdeal, cl::NullRange, globalSize, localSize, NULL, &event_halfstep);
  event_halfstep.wait();

}

void Visc::updateGlobMem( const cl_real & tau, const int & half_step , const cl::NDRange & globalSize, const cl::NDRange & localSize )
{

  cl::NDRange globalSize0 = Size;
  cl::NDRange localSize0 = cl::NullRange;

  //update u0 = u1 first
  int Nreg = 0;

  int regLevel = 0;

#ifdef VISCOUSON
  while( true ){
    cl::Buffer d_Nreg= cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_real), & Nreg); //global
    cl::Event event_updateViscGlobalMem;
    kernel_globVisc.setArg( 0, d_pi0 );
    kernel_globVisc.setArg( 1, d_pi1 );
    kernel_globVisc.setArg( 2, d_Umu0 );
    kernel_globVisc.setArg( 3, d_Umu1 );
    kernel_globVisc.setArg( 4, d_Newpi );
    kernel_globVisc.setArg( 5, d_Ed );
    kernel_globVisc.setArg( 6, d_Nreg );
    kernel_globVisc.setArg( 7, tau );
    kernel_globVisc.setArg( 8, half_step );
    kernel_globVisc.setArg( 9, Size );
    queue.enqueueNDRangeKernel( kernel_globVisc, cl::NullRange, globalSize0, localSize0, NULL, &event_updateViscGlobalMem );
    event_updateViscGlobalMem.wait();

    queue.enqueueReadBuffer( d_Nreg, CL_TRUE, 0, sizeof(cl_real), &Nreg );

//    break;
    if(Nreg==0 || regLevel>9) break;

    std::cout<<"Nreg="<<Nreg<<std::endl;
    std::cout<<"regLevel="<<regLevel<<std::endl;

    cl::Event event_regulatePimn;
    kernel_regulatePimn.setArg( 0, d_pi0 );
    kernel_regulatePimn.setArg( 1, d_pi1 );
    kernel_regulatePimn.setArg( 2, d_Umu0 );
    kernel_regulatePimn.setArg( 3, d_Umu1 );
    kernel_regulatePimn.setArg( 4, d_Newpi );
    kernel_regulatePimn.setArg( 5, d_Ed );
    kernel_regulatePimn.setArg( 6, d_Nreg );
    kernel_regulatePimn.setArg( 7, tau );
    kernel_regulatePimn.setArg( 8, half_step );
    kernel_regulatePimn.setArg( 9, Size );
    queue.enqueueNDRangeKernel( kernel_regulatePimn, cl::NullRange, globalSize0, localSize0, NULL, &event_regulatePimn );
    event_regulatePimn.wait();

    regLevel++;
    Nreg = 0;
  }



#endif

  //then update u1

  kernel_globIdeal.setArg( 0, d_Hm0 );
  kernel_globIdeal.setArg( 1, d_Tm0 );
  kernel_globIdeal.setArg( 2, d_Umu1 );
  kernel_globIdeal.setArg( 3, d_Ed );
  kernel_globIdeal.setArg( 4, d_NewTm00 );
  kernel_globIdeal.setArg( 5, d_NewUmu );
  kernel_globIdeal.setArg( 6, d_NewEd );
  kernel_globIdeal.setArg( 7, tau );
  kernel_globIdeal.setArg( 8, half_step );
  kernel_globIdeal.setArg( 9, Size );

  cl::Event event_update1;
  queue.enqueueNDRangeKernel( kernel_globIdeal, cl::NullRange, globalSize0, localSize0, NULL, &event_update1);
  event_update1.wait();

}

void Visc::runKernelEvolve()
{
  try{
    int LSZ = BSZ - 4;
    cl::NDRange globalSize1 = cl::NDRange( NX, NY, NZ );
    cl::NDRange  localSize1 = cl::NDRange( LSZ, LSZ, LSZ );
    cl_real tau = TAU0 ;

    for(int n=0; n<=30; n++){
      //output( tau, n, 20, 1, 1, 1 );
      output( tau, n, NTSKIP, NXSKIP, NYSKIP, NZSKIP );
      //bulkInfo( tau, n, 30 );
      std::cout<<"#tau="<<tau<<std::endl;

#ifdef VISCOUSON
      runKernelVisc( tau, 1, globalSize1, localSize1 );
#endif
      runKernelIdeal( tau, 1, globalSize1, localSize1 );
      //updateGlobMem(  tau, 1, globalSize1, localSize1 );
      /** Full step for pimn and d_Src from shear part */

#ifdef VISCOUSON
      runKernelVisc( tau, 0, globalSize1, localSize1 );
#endif
      runKernelIdeal( tau, 0, globalSize1, localSize1 );
      updateGlobMem(  tau, 0, globalSize1, localSize1 );

      tau = TAU0 + (n+1) * DT;
    }
    std::cout<<"#Evol finished\n";

  }                               
  catch(cl::Error &err){          
    std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
  }


}


//////////////////////  Output and get bulk information ////////////////////////////
void Visc::output( const cl_real & tau, const int & n, \
    const int & ntskip, const int & nxskip, const int & nyskip, const int & nzskip )
{
  /** \breif Get ED_MAX to see if ED_MAX is smaller than edec */

  cl_real EdMax = getEdMax();
  std::cout<<"#ED_MAX = "<< EdMax <<std::endl;

  cl_real Edec = Eos->Ed( TFRZ, 0.0 );
  bool Finished = ( EdMax < Edec );

  /** If n%nskip || EDMAX<edec, calculate the freeze out hyper surface */
  if( n % ntskip == 0 || Finished ){

    queue.enqueueReadBuffer( d_Ed, CL_TRUE, 0, Size*sizeof(cl_real), h_Ed.data() );
    queue.enqueueReadBuffer( d_Umu1, CL_TRUE, 0, Size*sizeof(cl_real4), h_Umu1.data() );

    /** Ed(tau) at middle point */
    std::stringstream fName;
    fName<<fPathOut<<"/Ttau.dat";
    std::ofstream f_Ttau( fName.str(), std::ios::out | std::ios::app );
    if( !f_Ttau.is_open() ) std::cerr<<"Error in opening Ttau.dat\n";
    else{
      if( n==0 ) f_Ttau<<"#tau   ed "<<std::endl;
      f_Ttau<<tau<<' '<<h_Ed.at( Size/2 )<<std::endl;
      f_Ttau.close();
    }

    /** Info for freeze out hyper surface calculation */
    std::stringstream fNameFrz ;
    fNameFrz<<fPathOut<<"/FrzInfo.dat";
    std::ofstream f_Frz( fNameFrz.str(), std::ios::out | std::ios::app );
    if( !f_Frz.is_open() ) std::cerr<<"Error in opening Ttau.dat\n";
    else{
      /* Edec, DT, xlo, NXF, nxskip*DX, ylo, NYF, nyskip*DY, zlo, NZF, nzskip*DZ */
      if( n==0 ){
        f_Frz<<"#Edec, DT, xlo, nx, dx, ylo, ny, dy, zlo, nz, dz \n";
        f_Frz<<Edec<<' '<<DT<<' '<<-NX/2*DX<<' '<<(NX-1)/nxskip+1 <<' '<< nxskip*DX<<' ' \
          <<' '<<-NY/2*DY<<' '<<(NY-1)/nyskip+1 <<' '<< nyskip*DY<<' ' \
          <<' '<<-NZ/2*DZ<<' '<<(NZ-1)/nzskip+1 <<' '<< nzskip*DZ<<'\n' ;
      }
      f_Frz<<tau<<' '<< n <<std::endl;
      f_Frz.close();
    }


    /** ED, vx, vy, vz 4D data */
    //char fname[256];
    //sprintf( fname, "edvxyz_%d.dat", n );

    std::stringstream fNameBulk;
    fNameBulk<<fPathOut<<"/edvxyz_"<<n<<".dat";
    std::ofstream fout( fNameBulk.str() );

    std::stringstream fBulk;

#ifdef VISCOUSON
    queue.enqueueReadBuffer( d_pi1,  CL_TRUE, 0, 10*Size*sizeof(cl_real), h_pi0.data() );
    fBulk<<"#tau  x   y   etas    Ed  Ut  Ux  Uy pitt pitx pity pixx pixy piyy pizz\n";
#else
    fBulk<<"#Ed(tau="<<tau<<")"<< "vx vy vz \n";
#endif

    for(int k=0; k<NX ; k += nxskip )
      for(int l=0; l<NY ; l += nyskip )
        for(int m=0; m<NZ ; m += nzskip )
        {
          int i = k*NY*NZ + l*NZ + m;
          cl_real4 u = h_Umu1.at(i);
#ifdef VISCOUSON
          // Gubser solution for viscous hydro has Temperature evolution
          //fout<<h_Ed.at(i)<<' '<<u.s[0]<<' '<<u.s[1]<<' '<<u.s[2]<<' '<<tau * u.s[3]<<'\n';
          fBulk<<tau<<' '<<(k-NX/2)*DX<<' '<<(l-NY/2)*DY<<' '<<(m-NZ/2)*DZ<<' '<<h_Ed.at(i)<<' '<<u.s[0]<<' '<<u.s[1]<<' '<<u.s[2]<<' '<<u.s[3]<<' ';
          fBulk<<h_pi0.at(10*i+0)<<' '<<h_pi0.at(10*i+1)<<' '<<h_pi0.at(10*i+2)<<' '<<h_pi0.at(10*i+3)<<' '<<h_pi0.at(10*i+4)<<' '<<h_pi0.at(10*i+5)<<' '<<h_pi0.at(10*i+6)<<' '<<h_pi0.at(10*i+7)<<' '<<h_pi0.at(10*i+8)<<' '<<h_pi0.at(10*i+9)<<'\n';

#else          
          fout<<h_Ed.at(i)<<' '<<u.s[1]/u.s[0]<<' '<<u.s[2]/u.s[0]<<' '<<tau * u.s[3]/u.s[0]<<'\n';
          //fBulk<<h_Ed.at(i)<<' '<<u.s[0]<<' '<<u.s[1]<<' '<<u.s[2]<<' '<<tau * u.s[3]<<'\n';
#endif
        }

    if( !fout.is_open() )std::cerr<<"# fout for ed umu pimn is not open\n";
    else{
      fout<<fBulk.str();
      fout.close();
    }


#ifdef VISCOUSON
    //    /** pimn */
    //    std::stringstream fNamePi;
    //    fNamePi<<fPathOut<<"/pimn_"<<n<<".dat";
    //    std::ofstream fout1( fNamePi.str() );
    //    if( !fout1.is_open() ) std::cerr<<"Error in opening edvxyz*.dat\n";
    //    else{
    //      fout1<<"#pimn(tau="<<tau<<"):"<< "pitt, pitx, pity, pitz, pixx, pixy, pixz, piyy, piyz, pizz\n";
    //      for(int k=0; k<NX ; k += nxskip )
    //        for(int l=0; l<NY ; l += nyskip )
    //          for(int m=0; m<NZ ; m += nzskip )
    //          {
    //            int i = k*NY*NZ*10 + l*NZ*10 + m*10;
    //            for(int n=0; n<10; n++ ){
    //              fout1<<h_pi0.at(i+n)<<' ';
    //            }
    //            fout1<<std::endl;
    //          }
    //      fout1.close();
    //    }
#endif

  }

  if( Finished ) {
    std::cout<<"#EdMax < EdFrz. Hydro stop..... \n";
    std::exit( 0 );
  }


}

/** Get t, x, y, ed, T, vx, vy, frc, medim density table for x in [-9.75, 9.75], y in [-9.75, 9.75] */
inline void Visc::bulkInfo( const cl_real & tau, const int & n, const int & nskip )
{
  if( n % nskip == 0 ){ 
    cl_real xmin = -9.75; 
    cl_real ymin = -9.75; 
    cl_real xmax = 9.75;
    cl_real ymax = 9.75;
    cl_int  Nx = 66;
    cl_int  Ny = 66;
    cl_int  Length = Nx * Ny;
    cl::Buffer d_bulkInfo = cl::Buffer( context, CL_MEM_READ_WRITE, Length*sizeof(cl_real4) ); //global

    cl::NDRange globalSize = cl::NDRange( Nx, Ny );
    cl::NDRange localSize =  cl::NullRange;

    kernel_bulk.setArg( 0, d_bulkInfo );
    kernel_bulk.setArg( 1, d_Ed );
    kernel_bulk.setArg( 2, d_Umu1 );
    kernel_bulk.setArg( 3, tau );
    kernel_bulk.setArg( 4, Length );

    cl::Event event_bulk;
    queue.enqueueNDRangeKernel( kernel_bulk, cl::NullRange, globalSize, localSize, NULL, &event_bulk); 
    event_bulk.wait();

    std::vector< cl_real4 > bulk( Length );
    queue.enqueueReadBuffer( d_bulkInfo, CL_TRUE, 0, Length*sizeof(cl_real4), bulk.data() );

    cl_real4 EdTVxVy;

    /** ED, vx, vy, vz 4D data */
    char fname[256];
    //sprintf( fname, "bulk%d.dat", n );
    //std::ofstream fout( fname, std::ios::out );
    //
    std::stringstream fName;
    fName<<fPathOut<<"/bulk.dat";
    std::ofstream fout( fName.str(), std::ios::app );
    if( fout.is_open() ) {
      //fout<<"#Ed(tau="<<tau<<")"<< "vx vy vz \n";
      for( int i=0; i!=Nx; i++ )
        for( int j=0; j!=Ny; j++ ){
          EdTVxVy = bulk.at( i*Ny + j );
          cl_real Tem = EdTVxVy.s[1] ;
          cl_real frac;
          if( Tem < 0.184 ) frac = 0.0;
          else if( Tem > 0.220 ) frac = 1.0;
          else frac = ( Tem - 0.184 ) / ( 0.220 - 0.184 );
          fout<<tau<<' '<<xmin+i*0.3<<' '<<ymin+0.3*j<<' '<<EdTVxVy.s[0]<<' '<<EdTVxVy.s[1] \
            <<' '<<EdTVxVy.s[2]<<' '<<EdTVxVy.s[3] \
            <<' '<<frac<<' '<<0.0<<'\n';
        }

      fout.close();
    }
    else{
      std::cerr<<"Error in opening bulk*.dat\n";
    }

  }
}




void Visc::testResults()
{
};

void Visc::clean()
{
};


