#include<cl_spec.h>
#include<ctime>
#include<cstdlib>

/////////////////////////////////////////////////
int main(int argc, char** argv)
{
    ///////////// Read Particles from particle data table //////////
    //
    Spec spec;
    
    char particleDataTable[256] = "../Resource/pdg05.dat";
    spec.ReadParticles( particleDataTable );

    std::cout<<"stable particle: \n";
    for( int i=0; i<spec.particles.size(); i++ ){
        if( spec.particles.at(i).stable == true ) std::cout<<spec.particles.at(i).monval<<' ';
    }
    std::cout<<'\n';

    std::string pathin;
    cl_real Tfrz = 0.137;
    if ( argc == 2 ) {
        pathin = std::string(argv[1]);
    }

    std::stringstream hypsfDataFile;
    std::stringstream pathout;
    pathout<<pathin;
    hypsfDataFile<<pathin<<"/hypersf.dat";
    // hypsfDataFile stores comments in the first row, Tfrz in the second row
    // dS^{0} dS^{1} dS^{2} dS^{3} vx vy veta eta_s for all other rows
    spec.ReadHyperSF(hypsfDataFile.str());

#ifdef VISCOUS_ON
    // pisfDataFile stores comments in the first row, 1.0/(2.0*T^2(e+P)) in the second row
    // pi^{00} 01 02 03 11 12 13 22 23 33 on the freeze out hyper surface for other rows
    std::stringstream pisfDataFile;
    pisfDataFile<<pathin<<"/pimnsf.dat";
    spec.ReadPimnSF(pisfDataFile.str());
#endif

    spec.SetPathOut( pathout.str() );

    //////////// Sample particles from SF //////////////////////////
    spec.SetTfrz( Tfrz );

    bool switch_off_decay = SWITCH_OFF_DECAY;

    std::cout << "begin to calc spec" << std::endl;
    spec.CalcSpec(switch_off_decay);

    //spec.ReadSpec();

    if ( !switch_off_decay ) {
      spec.ResoDecay();
    }

    spec.testResults();

    ///////////  Conect to UrQMD /////////////////////////////////

    return 0;
}
