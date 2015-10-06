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
    if ( argc == 3 ) {
        pathin = std::string(argv[1]);
        Tfrz = atof(argv[2]);
    }

    std::stringstream hypsfDataFile;
    std::stringstream pathout;
    pathout<<pathin;
    hypsfDataFile<<pathin<<"/Hypersf.dat";

    spec.ReadHyperSF( hypsfDataFile.str() );

    spec.SetPathOut( pathout.str() );

    //////////// Sample particles from SF //////////////////////////
    spec.SetTfrz( Tfrz );

    bool switch_off_decay = SWITCH_OFF_DECAY;

    spec.CalcSpec(switch_off_decay);

    //spec.ReadSpec();

    if ( !switch_off_decay ) {
      spec.ResoDecay();
    }

    spec.testResults();

    ///////////  Conect to UrQMD /////////////////////////////////

    return 0;
}
