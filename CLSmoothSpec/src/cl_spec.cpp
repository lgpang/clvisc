#include <cl_spec.h>
#include <cmath>

#define NBlocks 64


/*! \breif Kham summation to reduce accumate error */
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

/** get the kernel excution time in units of seconds */
cl_real excutionTime( cl::Event & event )
{
    cl_ulong tstart, tend; 
    event.getProfilingInfo( CL_PROFILING_COMMAND_START, & tstart ) ;
    event.getProfilingInfo( CL_PROFILING_COMMAND_END, & tend ) ;
    //std::cout<<"#run time="<<(tend - tstart )/1000<<"ms\n";
    return ( tend - tstart ) * 1.0E-9 ;
}


Spec::Spec()
{
    DataPath = "../results/";
}


Spec::~Spec()
{
}


cl::Context Spec::CreateContext( const cl_int & device_type )
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get( &platforms );
    if( platforms.size() == 0 ){
        std::cerr<<"no platform found\n";
        exit(-1);
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
        exit( -1 );
    }
}

void Spec::AddProgram( const char * fname)
{ //// An compact way to add program from file
    std::ifstream kernelFile( fname );
    if( !kernelFile.is_open() ) std::cerr<<"Open "<<fname << " failed!"<<std::endl;

    std::string sprog( std::istreambuf_iterator<char> (kernelFile), (std::istreambuf_iterator<char> ()) );
    cl::Program::Sources prog(1, std::make_pair(sprog.c_str(), sprog.length()));

    programs.push_back( cl::Program( context, prog ) );

}

void Spec::BuildPrograms( int i, const char * compile_options )
{ //// build programs and output the compile error if there is
    //for(std::vector<cl::Program>::size_type i=0; i!=programs.size(); i++)
    {
        try{
            programs.at(i).build(devices, compile_options);
        }
        catch(cl::Error & err){
            std::cerr<<err.what()<<"("<<err.err()<<")\n"<< programs.at(i).getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        }
    }

}

///////////////////////////////////////////////////////////////////////////////
void Spec::SetTfrz( const cl_real & TfrzIn )
{
    Tfrz = TfrzIn;
}




///////////////////////////////////////////////////////////////////////////////
void Spec::InitGrid(const cl_int & Nrapidity, const cl_real& Ylo, const cl_real & Yhi)
{
    const cl_real dY = ( Yhi - Ylo ) / (Nrapidity-1.0 );
    for( int i=0; i!=Nrapidity; i++ ){
        h_Y.push_back( Ylo + i * dY );
    }

    cl_real *pgauss, *wgauss;
    cl_real *pgaula, *wgaula;
    pgauss = gaulep48; wgauss=gaulew48;
    pgaula = gala15x;  wgaula=gala15w;

    double Phi[ NPHI ];
    for( int i=0; i<NPHI/2; i++ ){
        Phi[ NPHI-1-i ] = M_PI * ( pgauss[i] + 1.0 );
        Phi[ i ] = M_PI * ( 1.0 - pgauss[i] );
    }

    for( int i=0; i<NPHI; i++ ){
        //h_CSPhi.push_back( (cl_real2) { cos(Phi[i]), sin(Phi[i]) } );
        h_CPhi.push_back( cos(Phi[i]) );
        h_SPhi.push_back( sin(Phi[i]) );
    }

    const double ptlo = 0.0;
    const double invslope = 1.0/12.0;
    /** h_Pt: pt for stable particle 0-4 GeV; 
     *  Resonances: pt range 0-8 GeV; */
    for( int i=0; i<NPT; i++ ){
        h_Pt.push_back( invslope * pgaula[i] + ptlo );
    }

}

void Spec::ReadMuB( const std::string & dataFile )
{
    cl_int pid;
    cl_real chemicalpotential;
    std::ifstream fin( dataFile );
    if(fin.is_open()){
        while( fin.good() ){
            fin>>pid>>chemicalpotential;
            if( fin.eof() )break;  // eof() repeat the last line
            muB[ pid ] = chemicalpotential;
            std::cout<<"pid="<<pid<<" muB="<<muB[pid]<<std::endl;
        }
        fin.close();
    }
    else{
        std::cerr<<"#Can't open muB data file!\n";
        exit(0);
    }
}

///////////////////////////////////////////////////////////////////////////////
void Spec::ReadHyperSF( const std::string & dataFile )
{
    cl_real dA0, dA1, dA2, dA3, vx, vy, vh, tau, x, y, etas;
    std::ifstream fin(dataFile);
    char buf[256];
    if ( fin.is_open() ) {
        fin.getline(buf, 256);  // readin the comment
        std::string comments(buf);
        Tfrz = std::stof(comments.substr(7));
        while( fin.good() ){
            fin>>dA0>>dA1>>dA2>>dA3>>vx>>vy>>vh>>etas;
            if( fin.eof() )break;  // eof() repeat the last line
            if ( isnan(dA0) || isnan(dA1) || isnan(dA2) || isnan(dA3) ) {
                dA0 = 0.0; dA1 = 0.0; dA2 = 0.0; dA3=0.0;
                vx = 0.0; vy = 0.0; vh = 0.0; etas = 0.0;
                std::cout << "nan in hypersf data file!" << std::endl;
            }
            h_SF.push_back( (cl_real8){ dA0, dA1, dA2, dA3, vx, vy, vh, etas } );
        }
        fin.close();
        SizeSF = h_SF.size();
        std::cout<<"#hypersf size="<<SizeSF<<std::endl;
        std::cout<<"#Tfrz = "<< Tfrz << std::endl;
    }
    else{
        std::cerr<<"#Can't open hyper-surface data file!\n";
        exit(0);
    }
}

void Spec::ReadPimnSF(const std::string & piFile)
{
    std::ifstream fin2(piFile);
    char buf[256];
    cl_real pimn[10];
    if ( fin2.is_open() ) {
        fin2.getline(buf, 256);  // readin the comment
        std::string comments(buf);
        one_over_2TsqrEplusP = std::stof(comments.substr(20));
        while( fin2.good() ){
            for ( int i=0; i < 10; i++ ) {
                fin2 >> pimn[i];
            }
            if( fin2.eof() )break;  // eof() repeat the last line
            for ( int i=0; i < 10; i++ ) {
                if ( isnan(pimn[i]) ) pimn[i] = 0.0;
                h_pi.push_back(pimn[i]); 
            }
        }
        fin2.close();
    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for pimn!\n";
        exit(0);
    }

    if (SizeSF != h_pi.size()/10) {
        std::cout << "num of pi on sf is not correct!\n";
    }
}

void Spec::ReadParticles(char * particle_data_table)
{
    cl_real buff;
    char oneline[256];

    std::ifstream fin(particle_data_table);
    CParticle p;

    if( fin.is_open() ){
        while(fin.good()){
            p.stable = 0;
            fin>>p.monval>>p.name>>p.mass>>p.width         \
                >>p.gspin>>p.baryon>>p.strange>>p.charm     \
                >>p.bottom>>p.gisospin>>p.charge>>p.decays;

            p.antibaryon_spec_exists = 0;

            if(particle_data_table == "../Resource/resoweak.dat"){
                fin>>buff>>buff>>buff; 
            }//Not used in s95p-v1/pdg05.dat

            if( fin.eof() ) break;
            CDecay dec;

            for(int k=0; k<p.decays; k++){
                fin>>dec.pidR>>dec.numpart>>dec.branch>>dec.part[0] \
                    >>dec.part[1]>> dec.part[2]>> dec.part[3]>> dec.part[4];

                decay.push_back( dec );
            }
            //if(nump1)p.stable=1;
            if(p.width < 1.0E-8)p.stable=1;
            particles.push_back(p);

        }
        fin.close();
    }
    else{
        std::cerr<<"#Failed to open pdg data table\n";
        exit(0);
    }


    CParticle antiB;//anti-baryon
    int N=particles.size();
    for(std::vector<CParticle>::size_type i=0; i!=N; i++){
        /** If unstable, pt range 0-8; if stable, pt range 0-4 */
        //cl_real resizePtRange = particles[i].stable ? 1.0 : 1.0 ;
        cl_real chem = muB[ particles[i].monval ];
        h_HadronInfo.push_back( (cl_real4){particles[i].mass, particles[i].gspin, particles[i].baryon?1.0f:-1.0f, chem} );

        if( particles[i].baryon ){
            antiB.monval = -particles[i].monval;
            antiB.name   = "A";
            antiB.name.append(particles[i].name);
            antiB.mass = particles[i].mass;
            antiB.width = particles[i].width;
            antiB.gspin = particles[i].gspin;
            antiB.baryon = -particles[i].baryon;
            antiB.strange= -particles[i].strange;
            antiB.charm = -particles[i].charm;
            antiB.bottom = -particles[i].bottom;
            antiB.gisospin=particles[i].gisospin;
            antiB.charge=-particles[i].charge;
            antiB.decays=particles[i].decays;
            antiB.stable=particles[i].stable;
            antiB.antibaryon_spec_exists = 1;
            particles.push_back(antiB);

            h_HadronInfo.push_back( (cl_real4){antiB.mass, antiB.gspin, antiB.baryon?1.0f:-1.0f, chem} );
        }
    }

    SizePID = h_HadronInfo.size();

    for( int i =0; i< SizePID; i++ ){
        newpid[ particles[i].monval ] =  i;
    }

    std::cout<<"newpid of pion = "<<newpid[ 211 ]<<std::endl;
    std::cout<<"newpid of proton = "<<newpid[ 2212 ]<<std::endl;
    std::cout<<"newpid of -13334 = "<<newpid[ -13334 ]<<std::endl;

    //    for( int i = 0; i< decay.size(); i++ ){
    //        std::cout<<"decay: "<<decay.at(i).pidR<<"->"<<decay.at(i).part[0] <<"+" \
    //            <<decay.at(i).part[1] <<"+" \
    //            <<decay.at(i).part[2] <<"+" \
    //            <<decay.at(i).part[3] <<"\n" ;
    //
    //    }

}


void Spec::initializeCL()
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
        std::string dev_vendor = devices[0].getInfo<CL_DEVICE_VENDOR>();
        std::cout<<"#dev_vendor="<<dev_vendor<<std::endl;

        int LenVector = devices[0].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
        std::cout<<"#preferred vector width float ="<< LenVector << std::endl;

        compile_options << "-I../src"<<" ";
        //compile_options << "-cl-nv-verbose"<<" ";
        //compile_options << "-cl-fast-relaxed-math -DMAC"<<" ";
        //compile_options << "-cl-opt-disable"<<" ";

        if ( sizeof(cl_real) == 4 ){
            compile_options << "-D USE_SINGLE_PRECISION"<<" ";
        }

        compile_options << "-D NBlocks="<<NBlocks<<" ";       
        compile_options << "-D BSZ="<<BSZ<<" ";       
        compile_options << "-D SizeSF="<<SizeSF<<" "; 
        compile_options << "-D SizePID="<<SizePID<<" "; 
        compile_options << "-D NY="<<NY<<" ";       
        compile_options << "-D NPT="<<NPT<<" ";    
        compile_options << "-D NPHI="<<NPHI<<" "; 
        compile_options << "-D YLO="<<YLO<<" ";  
        compile_options << "-D YHI="<<YHI<<" "; 
        compile_options << "-D INVSLOPE="<<INVSLOPE<<" ";   
        compile_options << "-D Tfrz="<<Tfrz<<" ";     
#ifdef VISCOUS_ON        
        compile_options << "-D VISCOUS_ON" << " ";
        compile_options << "-D TCOEFF=" << one_over_2TsqrEplusP << " ";
        std::cout << "TFRZ=" << Tfrz << " TCOEFF=" << one_over_2TsqrEplusP << std::endl;
#endif
        queue = cl::CommandQueue( context, devices[0], CL_QUEUE_PROFILING_ENABLE );
        //queue = cl::CommandQueue( context, devices[1], CL_QUEUE_PROFILING_ENABLE );
        //std::cout<<"Vendor="<<dev_vendor<<'\n';

#ifdef  LOEWE_CSC
        AddProgram( "../src/kernel_spec_csc.cl" );
#else
        AddProgram( "../src/kernel_spec.cl" );
#endif


        AddProgram( "../src/kernel_decay.cl" );

        BuildPrograms(0, compile_options.str().c_str() );

        //compile_options << "-cl-opt-disable"<<" ";
        BuildPrograms(1, compile_options.str().c_str() );

        kernel_subspec = cl::Kernel( programs.at(0), "get_sub_dNdYPtdPtdPhi" );
        kernel_spec = cl::Kernel( programs.at(0), "get_dNdYPtdPtdPhi" );

        kernel_decay2 = cl::Kernel( programs.at(1), "decay_2body" );
        kernel_decay3 = cl::Kernel( programs.at(1), "decay_3body" );
        kernel_sumdecay = cl::Kernel( programs.at(1), "sumResoDecay" );

        InitGrid( NY, -8.0, 8.0 );

        h_Spec.resize( NY*NPT*NPHI );

        d_SF = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real8), h_SF.data()); //global memory

#ifdef  VISCOUS_ON
        d_pi = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 10*SizeSF*sizeof(cl_real), h_pi.data()); //global memory
#endif

        d_HadronInfo = cl::Buffer( context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, SizePID*sizeof(cl_real4) , h_HadronInfo.data()); //constant memory

        d_Y = cl::Buffer( context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, NY*sizeof(cl_real) , h_Y.data()); //constant memory
        d_Pt = cl::Buffer( context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, NPT*sizeof(cl_real) , h_Pt.data()); //constant memory
        d_CPhi = cl::Buffer( context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, NPHI*sizeof(cl_real) , h_CPhi.data()); //constant memory
        d_SPhi = cl::Buffer( context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, NPHI*sizeof(cl_real) , h_SPhi.data()); //constant memory

        d_SubSpec = cl::Buffer( context, CL_MEM_READ_WRITE, NY*NPT*NPHI*NBlocks*sizeof(cl_real) ); //global memory
        d_Spec = cl::Buffer( context, CL_MEM_READ_WRITE, SizePID*NY*NPT*NPHI*sizeof(cl_real) ); //global memory
    }
    catch (cl::Error & err ){
        std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
    }

}



void Spec::SetPathOut( const std::string & path )
{
    DataPath = path;
    std::cout<<"path="<<DataPath<<std::endl;
}


void Spec::ReadSpec()
{
    char fname[256];
    cl_real temp;

    try{
        initializeCL();
        for( int pid=0; pid<SizePID; pid++ ){
            sprintf( fname, "%s/dNdYPtdPtdPhi_%d.dat", DataPath.c_str(), pid );
            std::ifstream fin( fname );
            if( fin.is_open() ){
                for(int k = 0; k<NY*NPT*NPHI; k++ ){
                    fin>>temp;
                    h_Spec.at(k) = temp;
                }
            }
            else{
                std::cerr<<"#failed to open spec for read \n";
                exit(0);
            }
            queue.enqueueWriteBuffer( d_Spec, CL_TRUE, pid*NY*NPT*NPHI*sizeof(cl_real), NY*NPT*NPHI*sizeof(cl_real), h_Spec.data() );
            std::cout<<"#pid "<<pid<<" read in \n";
        }
    }
    catch(cl::Error &err){          
        std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
    }

}




namespace {
    /** Write spectra dNdYptdptdphi or dNdEtaptdptdphi to file */
    void WriteSpec(std::string fname, cl_real spec[NY][NPT][NPHI]){
        std::ofstream fspec(fname);
        if ( fspec.is_open() ) {
            for( int i=0; i<NY; i++)
                for( int j=0; j<NPT; j++)
                    for( int k=0; k<NPHI; k++)
                        fspec<<spec[i][j][k]<<' ';
        } else {
            std::cerr << "#failed to open " << fname << " for writing \n ";
        }
        fspec.close();
    }

} // end unnamed namespace

/** Calc thermal spec for all the hadrons */
void Spec::CalcSpec(bool switch_off_decay)
{
    try {
        /** store spec in Y,Pt,Phi */
        cl_real dNdYPtdPtdPhi[NY][NPT][NPHI];
        /** store spec in Eta,Pt,Phi */
        cl_real dNdEtaPtdPtdPhi[NY][NPT][NPHI];

        initializeCL();

        cl_int BlockSize = BSZ;
        cl_int Size = BlockSize * NBlocks;
        cl::NDRange globalSize = cl::NDRange( Size );
        //localSize =  cl::NullRange;   //system will choose a best localSize
        cl::NDRange localSize = cl::NDRange( BlockSize );

        cl::NDRange globalSize1 = cl::NDRange(  NY * NPT * NPHI*NBlocks );
        cl::NDRange localSize1  = cl::NDRange( NBlocks );

        char fname[256];
        for(int pid=0; pid!=SizePID; pid++){
            cl_int monval = particles[pid].monval;

            // if switch_off_decay, only calc direct spec for pi, K, proton
            if ( switch_off_decay ) {
                if ( !(monval == 211 || monval == 321 || monval == 2212
                            || monval == 999) ) continue;
                // 999 is D0
            }

            if ( ! particles[pid].antibaryon_spec_exists ) {
#ifdef VISCOUS_ON
                kernel_subspec.setArg( 0, d_SF );
                kernel_subspec.setArg( 1, d_pi );
                kernel_subspec.setArg( 2, d_SubSpec );
                kernel_subspec.setArg( 3, d_HadronInfo );
                kernel_subspec.setArg( 4, d_Y );
                kernel_subspec.setArg( 5, d_Pt );
                kernel_subspec.setArg( 6, d_CPhi );
                kernel_subspec.setArg( 7, d_SPhi );
                kernel_subspec.setArg( 8, pid );
#else

                kernel_subspec.setArg( 0, d_SF );
                kernel_subspec.setArg( 1, d_SubSpec );
                kernel_subspec.setArg( 2, d_HadronInfo );
                kernel_subspec.setArg( 3, d_Y );
                kernel_subspec.setArg( 4, d_Pt );
                kernel_subspec.setArg( 5, d_CPhi );
                kernel_subspec.setArg( 6, d_SPhi );
                kernel_subspec.setArg( 7, pid );
#endif
                /** Because AMD GPUs can not use big array in private memory.
                 * The dNdYPtdPtdPhi[41*15*48] in the kernel are replaced by
                 * dNdYPtdPtdPhi[48] with only phi values stored privately
                 * in order to run on LOEWE_CSC */
                double excution_time_step1 = 0.0;
#ifdef  LOEWE_CSC
                for(int id_Y_Pt_Phi=0; id_Y_Pt_Phi<NY*NPT; id_Y_Pt_Phi++){
                    cl::Event event;

#ifdef VISCOUS_ON
                    kernel_subspec.setArg(9, id_Y_Pt_Phi);
#else
                    kernel_subspec.setArg(8, id_Y_Pt_Phi);
#endif
                    queue.enqueueNDRangeKernel(kernel_subspec, cl::NullRange, \
                            globalSize, localSize, NULL, &event);
                    event.wait();
                    excution_time_step1 += excutionTime(event);
                }
#else
                cl::Event event;
                queue.enqueueNDRangeKernel( kernel_subspec, cl::NullRange, \
                        globalSize, localSize, NULL, &event);
                event.wait();
                excution_time_step1 += excutionTime(event);
#endif

                kernel_spec.setArg(0, d_SubSpec);
                kernel_spec.setArg(1, d_Spec);
                kernel_spec.setArg(2, pid);


                std::cout<<"#hadron mass:"<<h_HadronInfo.at(pid).s[0]<<std::endl;
                std::cout<<"#hadron gspin:"<<h_HadronInfo.at(pid).s[1]<<std::endl;
                std::cout<<"#hadron fb:"<<h_HadronInfo.at(pid).s[2]<<std::endl;
                std::cout<<"#hadron resizePtRange:"<<h_HadronInfo.at(pid).s[3]<<std::endl;

                cl::Event event1;
                queue.enqueueNDRangeKernel(kernel_spec, cl::NullRange, \
                        globalSize1, localSize1, NULL, &event1);
                event1.wait();

                std::cout<<"#reduction step 1 costs: "<<excution_time_step1<<std::endl;
                std::cout<<"#reduction step 2 costs: "<<excutionTime(event1)<<std::endl;
                std::cout<<"#pid: "<<particles[pid].monval << " id=" << pid << std::endl;
            } else {
                /// if antibaryon_spec_exists == 1, copy spec
                int antibaryon_pid = newpid[-particles[pid].monval];
                std::cout << "pid=" << pid << " antipid=" << antibaryon_pid << std::endl;

                /// in order command queue, do not need event.wait()
                /// copy spectra from its anti particle
                cl::Event event2;
                queue.enqueueCopyBuffer(d_Spec, d_Spec, \
                        antibaryon_pid*NY*NPT*NPHI*sizeof(cl_real), \
                        pid*NY*NPT*NPHI*sizeof(cl_real), \
                        NY*NPT*NPHI*sizeof(cl_real), NULL, &event2);
                event2.wait();
            }

            // Only output pion, kaon and proton before resonance decay
            if ( monval == 211 || monval == 321 || monval == 2212 || monval == 999) {
                queue.enqueueReadBuffer(d_Spec, CL_TRUE, pid*NY*NPT*NPHI*sizeof(cl_real),
                        NY*NPT*NPHI*sizeof(cl_real), h_Spec.data() );
                for ( int i = 0; i < NY; i++ ) 
                    for ( int j = 0; j < NPT; j++ ) 
                        for ( int k = 0; k < NPHI; k++ ) {
                            dNdYPtdPtdPhi[i][j][k] = h_Spec.at(i*NPT*NPHI+j*NPHI+k);
                        }
                // write spec in Y, Pt, Phi
                sprintf( fname, "%s/dNdYPtdPtdPhi_%d.dat", DataPath.c_str(), monval );
                WriteSpec(fname, dNdYPtdPtdPhi);

                // write spec in Eta, Pt, Phi
                cl_real mass = particles.at(pid).mass;
                get_dNdEtaPtdPtdPhi(mass, dNdYPtdPtdPhi, dNdEtaPtdPtdPhi);
                sprintf( fname, "%s/dNdEtaPtdPtdPhi_%d.dat", DataPath.c_str(), monval );
                WriteSpec(fname, dNdEtaPtdPtdPhi);
            }
        }
    }
    catch(cl::Error &err){          
        std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
    }
}

/** \breif calc the norm3 factor for 3 body decay */
cl_real norm3int( const cl_real & x, const cl_real & a, const cl_real & b, \
        const cl_real & c, const cl_real & d )
{
    return sqrt( (a-x)*(b-x)*(x-c)*(x-d) ) / x;
}

cl_real norm3( const cl_real & mr, const cl_real & a, const cl_real & b, \
        const cl_real & c, const cl_real & d )
{
    int n=12;
    cl_real	p[] = {	0.9815606342,	0.9041172563, 0.7699026741,	0.5873179542, 0.3678314989,	0.1252334085	};
    cl_real w[] = {	0.0471753363,	0.1069393259, 0.1600783285,	0.2031674267, 0.2334925365,	0.2491470458	};

    cl_real coef = mr*mr/(2.0*M_PI);

    cl_int  ix;
    cl_real xlo = c;
    cl_real xhi = b;
    cl_real xoffs = 0.5 * ( xlo + xhi );
    cl_real xdiff = 0.5 * ( xhi - xlo );
    cl_real s = 0;

    for( ix=0; ix<n/2; ix++ ) 	/* n is even */
        s += w[ix] * ( norm3int(xoffs+xdiff*p[ix], a, b, c, d) + 
                norm3int(xoffs-xdiff*p[ix], a, b, c, d) );
    s *=  xdiff ;

    return coef / std::max(s, (cl_real)1.0E-9);
}
////////////////////////////////////////////////////////////////////////

/** \param j  the index of decay channel in decay list 
 *  \param k  the position of daughter particle icay products */
void Spec::AddReso(cl_int pidR, cl_int j, cl_int k, \
        std::vector<cl_real> & branch, std::vector<cl_real4> & mass , \
        std::vector<cl_int>  & resoNum,std::vector<cl_real > & h_norm3)
{

    cl_int4 m = (cl_int4) {   
        newpid[decay.at(j).part[0]],  
            newpid[decay.at(j).part[1]], 
            newpid[decay.at(j).part[2]], 
            newpid[decay.at(j).part[3]] };

    if( k==1 ) m = (cl_int4) { m.s[1], m.s[0], m.s[2], m.s[3] };
    if( k==2 ) m = (cl_int4) { m.s[2], m.s[0], m.s[1], m.s[3] };
    if( k==3 ) m = (cl_int4) { m.s[3], m.s[0], m.s[1], m.s[2] };

    cl_real mr, m1, m2, m3, m4, a, b, c, d;
    resoNum.push_back( pidR );
    branch.push_back( decay.at(j).branch );
    mr = particles[ pidR ].mass; 
    m1 = particles[ m.s[0] ].mass;
    m2 = particles[ m.s[1] ].mass;

    switch( abs(decay.at(j).numpart) ){
        case 1:
            break;
        case 2:
            while( (m1 + m2) > mr ){
                mr += 0.25 * particles[ pidR ].width;
                m1 -= 0.5  * particles[m.s[0]].width;
                m2 -= 0.5  * particles[m.s[1]].width;
            }
            mass.push_back( (cl_real4){mr, m1, m2, 0.0} );
            break;
        case 3:
            m3 = particles[ m.s[2] ].mass;  
            a = (mr + m1) * (mr + m1);
            b = (mr - m1) * (mr - m1);
            c = (m2 + m3) * (m2 + m3);
            d = (m2 - m3) * (m2 - m3);
            h_norm3.push_back( norm3(mr, a, b, c, d) );
            mass.push_back( (cl_real4){mr, m1, m2, m3} );
            break;
        case 4:
            m3 = particles[ m.s[2] ].mass;  
            m4 = particles[ m.s[3] ].mass;
            m3 = 0.5*( m3 + m4 + mr - m1 - m2 );
            a = (mr + m1) * (mr + m1);
            b = (mr - m1) * (mr - m1);
            c = (m2 + m3) * (m2 + m3);
            d = (m2 - m3) * (m2 - m3);
            h_norm3.push_back( norm3(mr, a, b, c, d) );
            mass.push_back( (cl_real4){mr, m1, m2, m3} );
            break;
        default:
            std::cout<<"#"<<abs(decay.at(j).numpart)<<" body decay is not implemented yet\n";
            exit(0);
    }
}

/** \breif Get decay information for nbody decay where 
 *  \param nbody 2,3,4
 *  \param branch branch ratio
 *  \param mass mr,m1,m2,m3
 *  \param resoNum pid for resonance 
 *  \param h_norm3 norm factor for 3 body decay */
void Spec::getDecayInfo( cl_int pid, cl_int nbody, \
        std::vector<cl_real> & branch, std::vector<cl_real4> & mass , \
        std::vector<cl_int>  & resoNum,std::vector<cl_real > & h_norm3)
{
    CDecay dec;
    cl_int pidR, pidAR;
    switch( particles[pid].baryon ){
        case 1:
            for( int j=0; j<decay.size(); j++ )
            {
                dec = decay.at(j);
                for( int k=0; k<abs(dec.numpart); k++ ){
                    if( (abs(dec.numpart) == nbody) && (newpid[dec.part[k]]==pid) ){
                        AddReso( newpid[dec.pidR], j, k, branch, mass, resoNum, h_norm3 );
                    }
                }
            }
            break;
        case -1:
            /** There is no antibaryon decay in decay table, use baryon decay info*/
            for( int j=0; j<decay.size(); j++ )
            {
                dec = decay.at(j);
                for( int k=0; k<abs(dec.numpart); k++ ){
                    if( (abs(dec.numpart) == nbody) && ( newpid[-dec.part[k]]==pid ) ){
                        AddReso( newpid[-dec.pidR], j, k, branch, mass, resoNum, h_norm3 );
                    }
                }
            }
            break;
        case 0:
            for( int j=0; j<decay.size(); j++ )
            {
                dec = decay.at(j);
                pidR =  newpid[ dec.pidR];
                for( int k=0; k<abs(dec.numpart); k++ ){
                    if( particles[ pidR ].baryon == 1){
                        pidAR = newpid[-dec.pidR];
                        if( (particles[pid].charge==0) && (particles[pid].strange==0) )
                        {
                            if( (abs(dec.numpart) == nbody) && ( newpid[dec.part[k]]==pid ) ){
                                AddReso( pidR, j, k, branch, mass, resoNum , h_norm3);
                                AddReso( pidAR, j, k, branch, mass, resoNum , h_norm3);
                            }
                        }
                        else{
                            if( (abs(dec.numpart) == nbody) && ( newpid[dec.part[k]]==pid ) ){
                                AddReso( pidR, j, k, branch, mass, resoNum , h_norm3);
                            }
                            if( (abs(dec.numpart) == nbody) && ( newpid[-dec.part[k]]==pid ) ){
                                AddReso( pidAR, j, k, branch, mass, resoNum , h_norm3);
                            }
                        }
                    }
                    else{
                        if( (abs(dec.numpart) == nbody) && ( newpid[dec.part[k]]==pid ) ){
                            AddReso( pidR, j, k, branch, mass, resoNum , h_norm3);
                        }
                    }
                }
            }
            break;
        default:
            std::cerr<<"Error in get particle's baryon number \n";
            exit(0);
    }

}


void Spec::ResoDecay()
{
    try{
        /** store spec in Y,Pt,Phi */
        cl_real dNdYPtdPtdPhi[NY][NPT][NPHI];
        /** store spec in Eta,Pt,Phi */
        cl_real dNdEtaPtdPtdPhi[NY][NPT][NPHI];

        /** store charged particle spec in Eta,Pt,Phi */
        cl_int num_of_charged_stable_hadrons = 0;
        cl_real dNdEtaPtdPtdPhi_Charged[NY][NPT][NPHI];
        for ( int i = 0; i < NY; i++ ) 
            for ( int j = 0; j < NPT; j++ ) 
                for ( int k = 0; k < NPHI; k++ ) {
                    dNdEtaPtdPtdPhi_Charged[i][j][k] = 0.0;
        }

        /** output file name for dNdY(Eta)PtdPtdPhi spec*/
        char fname[256];


        std::vector< cl_int > h_resoNum;
        std::vector< cl_real4 > h_mass;
        std::vector< cl_real > h_branch;
        std::vector< cl_real > h_norm3;


        cl::Buffer d_resoNum;
        cl::Buffer d_mass;
        cl::Buffer d_branch;
        cl::Buffer d_Decay;
        cl::Buffer d_norm3;
        int nchannels;

        cl_real decayTimeGpu = 0.0;

        for( int pid = SizePID-1; pid > 0; pid -- ){
            h_resoNum.clear();
            h_mass.clear();
            h_branch.clear();
            getDecayInfo( pid, 2, h_branch, h_mass, h_resoNum, h_norm3 );
            nchannels = h_resoNum.size() ;

            if( nchannels > 0 ){
                d_resoNum = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_int), h_resoNum.data() ); //global memory
                d_mass = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_real4), h_mass.data() ); //global memory
                d_branch = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_real), h_branch.data() ); //global memory
                d_Decay = cl::Buffer( context, CL_MEM_READ_WRITE , NY*NPT*NPHI*nchannels*sizeof(cl_real) ); //global memory

                kernel_decay2.setArg( 0, d_Spec );
                kernel_decay2.setArg( 1, d_Decay );
                kernel_decay2.setArg( 2, d_mass );
                kernel_decay2.setArg( 3, d_branch );
                kernel_decay2.setArg( 4, d_resoNum );
                kernel_decay2.setArg( 5, pid );
                cl::Event event;
                queue.enqueueNDRangeKernel( kernel_decay2, cl::NullRange, \
                        cl::NDRange(nchannels * BSZ), cl::NDRange(BSZ), NULL, &event);
                event.wait();

                std::cout<<"#"<<pid<<"2body decay costs: "<<excutionTime( event )<<std::endl;
                decayTimeGpu += excutionTime( event ) ; 

                kernel_sumdecay.setArg( 0, d_Decay );
                kernel_sumdecay.setArg( 1, d_Spec );
                kernel_sumdecay.setArg( 2, nchannels );
                kernel_sumdecay.setArg( 3, pid );

                cl::Event event1;
                queue.enqueueNDRangeKernel( kernel_sumdecay, cl::NullRange, \
                        cl::NDRange(NY*NPT*NPHI), cl::NullRange, NULL, &event1);
                event1.wait();
            }

            std::cout<<pid<<"# 2 body decay finished \n";


            h_resoNum.clear();
            h_mass.clear();
            h_branch.clear();
            h_norm3.clear();
            getDecayInfo( pid, 3, h_branch, h_mass, h_resoNum, h_norm3 );
            getDecayInfo( pid, 4, h_branch, h_mass, h_resoNum, h_norm3 );
            nchannels = h_resoNum.size() ;
            if( nchannels > 0 ){
                d_resoNum = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_int), h_resoNum.data() ); //global memory
                d_mass = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_real4), h_mass.data() ); //global memory
                d_branch = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_real), h_branch.data() ); //global memory
                d_norm3 = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nchannels*sizeof(cl_real), h_norm3.data() ); //global memory
                d_Decay = cl::Buffer( context, CL_MEM_READ_WRITE , NY*NPT*NPHI*nchannels*sizeof(cl_real) ); //global memory

                kernel_decay3.setArg( 0, d_Spec );
                kernel_decay3.setArg( 1, d_Decay );
                kernel_decay3.setArg( 2, d_mass );
                kernel_decay3.setArg( 3, d_branch );
                kernel_decay3.setArg( 4, d_norm3 );
                kernel_decay3.setArg( 5, d_resoNum );
                kernel_decay3.setArg( 6, pid );
                cl::Event event;
                queue.enqueueNDRangeKernel( kernel_decay3, cl::NullRange, \
                        cl::NDRange(nchannels * BSZ), cl::NDRange(BSZ), NULL, &event);
                event.wait();

                std::cout<<"#"<<pid<<"3body decay costs: "<<excutionTime( event )<<std::endl;
                decayTimeGpu += excutionTime( event ) ; 

                kernel_sumdecay.setArg( 0, d_Decay );
                kernel_sumdecay.setArg( 1, d_Spec );
                kernel_sumdecay.setArg( 2, nchannels );
                kernel_sumdecay.setArg( 3, pid );

                cl::Event event1;
                queue.enqueueNDRangeKernel( kernel_sumdecay, cl::NullRange, \
                        cl::NDRange(NY*NPT*NPHI), cl::NullRange, NULL, &event1);
                event1.wait();

            }

            std::cout<<pid<<" 3 body decay finished \n";

            if ( particles.at( pid ).stable ){
                queue.enqueueReadBuffer( d_Spec, CL_TRUE, pid*NY*NPT*NPHI*sizeof(cl_real), NY*NPT*NPHI*sizeof(cl_real), h_Spec.data() );
                cl_int monval = particles[pid].monval;
                for ( int i = 0; i < NY; i++ ) 
                    for ( int j = 0; j < NPT; j++ ) 
                        for ( int k = 0; k < NPHI; k++ ) {
                            dNdYPtdPtdPhi[i][j][k] = h_Spec.at(i*NPT*NPHI+j*NPHI+k);
                        }
                // write spec in Y, Pt, Phi
                if( monval > 0 ) sprintf( fname, "%s/dNdYPtdPtdPhi_Reso%d.dat", DataPath.c_str(), monval );
                else sprintf( fname, "%s/dNdYPtdPtdPhi_ResoA%d.dat", DataPath.c_str(), -monval );
                WriteSpec(fname, dNdYPtdPtdPhi);

                // write spec in Eta, Pt, Phi
                cl_real mass = particles.at(pid).mass;
                get_dNdEtaPtdPtdPhi(mass, dNdYPtdPtdPhi, dNdEtaPtdPtdPhi);
                if( monval > 0 ) sprintf( fname, "%s/dNdEtaPtdPtdPhi_Reso%d.dat", DataPath.c_str(), monval );
                else sprintf( fname, "%s/dNdEtaPtdPtdPhi_ResoA%d.dat", DataPath.c_str(), -monval );
                WriteSpec(fname, dNdEtaPtdPtdPhi);

                // Sum charged particle spectra
                if ( particles.at(pid).charge ) {
                    for ( int i = 0; i < NY; i++ ) 
                        for ( int j = 0; j < NPT; j++ ) 
                            for ( int k = 0; k < NPHI; k++ ) {
                                dNdEtaPtdPtdPhi_Charged[i][j][k] += dNdEtaPtdPtdPhi[i][j][k];
                    }

                    num_of_charged_stable_hadrons++;
                } // end if partile.charge == true
            } // end if partile.stable == true
        } // end for pid

        sprintf(fname, "%s/dNdEtaPtdPtdPhi_Charged.dat", DataPath.c_str());
        WriteSpec(fname, dNdEtaPtdPtdPhi_Charged);

        std::cout<<"#Num of charged hdarons ="<< num_of_charged_stable_hadrons << std::endl;

        std::cout<<"#All decay costs: "<<decayTimeGpu<<std::endl;
    }  catch(cl::Error &err) {
        std::cerr<<"Error:"<<err.what()<<"("<<err.err()<<")\n";
    }

}


/* Get dNdEtaPtdPtdPhi from dNdYPtdPtdPhi for particle with mass=mass 
 * dN/dEta = dN/dY*dY/dEta 
 * E = mt*cosh(Y), P = pt*cosh(eta)
 * mt*sinh(Y) = pt*shin(eta) = pz
 * dY/dEta = P/E 
 * */
void Spec::get_dNdEtaPtdPtdPhi(cl_real mass, cl_real dNdYPtdPtdPhi[NY][NPT][NPHI], \
        cl_real dNdEtaPtdPtdPhi[NY][NPT][NPHI]){
    for(int k=0; k!=NY; k++){
        cl_real eta=h_Y.at(k);
        for(int i=0; i!=NPT; i++){
            cl_real pt=h_Pt.at(i);
            cl_real mt=sqrt(pt*pt+mass*mass);
            cl_real ptce=pt*cosh(eta);
            cl_real dYdEta=ptce/sqrt(mass*mass+ptce*ptce);//   P/E  
            cl_real Y=atanh(dYdEta*tanh(eta));//  dY/deta=P/E

            if(Y<YLO){
                Y=YLO;
            } else if(Y>YHI){
                Y=YHI;
            }

            int IY=1;
            while(Y>h_Y.at(IY) && IY<NY-1){
                IY ++;
            }

            assert(IY>=0 && IY<=NY-1);
            for(int j=0; j!=NPHI; j++){
                double Y1= h_Y.at(IY-1);
                double Y2= h_Y.at(IY);
                double FY1=dNdYPtdPtdPhi[IY-1][i][j];
                double FY2=dNdYPtdPtdPhi[IY][i][j];
                double a=(Y-Y1)/(Y2-Y1);
                dNdEtaPtdPtdPhi[k][i][j] = ((1.0-a)*FY1+a*FY2)*dYdEta;
            }
        }
    }
}




void Spec::testResults()
{

    std::vector<cl_real> h_DA0( SizeSF );
    for( int i =0 ; i<SizeSF; i ++ ){
        h_DA0.at(i) = h_SF.at(i).s[0];
    }

    cl_real sumda0_cpu = reduceCPU( h_DA0.data(), SizeSF );

    std::cout<<"#sum DA0 on CPU = "<<sumda0_cpu<<std::endl;

};

void Spec::clean()
{
};

