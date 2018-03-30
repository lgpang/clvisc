#include<cl_spec.h>
#include<ctime>
#include<cstdlib>
#include<cmath>
//#include<omp.h>

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

    void get_dNdEtaPtdPtdPhi(const Spec & spec, cl_real mass, cl_real dNdYPtdPtdPhi[NY][NPT][NPHI], \
            cl_real dNdEtaPtdPtdPhi[NY][NPT][NPHI]){
        for(int k=0; k!=NY; k++){
            cl_real eta=spec.h_Y.at(k);
            for(int i=0; i!=NPT; i++){
                cl_real pt=spec.h_Pt.at(i);
                cl_real mt=std::sqrt(pt*pt+mass*mass);
                cl_real ptce=pt*std::cosh(eta);
                cl_real dYdEta=ptce/std::sqrt(mass*mass+ptce*ptce);//   P/E  
                cl_real Y=std::atanh(dYdEta*std::tanh(eta));//  dY/deta=P/E

                if(Y<YLO){
                    Y=YLO;
                } else if(Y>YHI){
                    Y=YHI;
                }

                int IY=1;
                while(Y>spec.h_Y.at(IY) && IY<NY-1){
                    IY ++;
                }

                assert(IY>=0 && IY<=NY-1);
                for(int j=0; j!=NPHI; j++){
                    double Y1= spec.h_Y.at(IY-1);
                    double Y2= spec.h_Y.at(IY);
                    double FY1=dNdYPtdPtdPhi[IY-1][i][j];
                    double FY2=dNdYPtdPtdPhi[IY][i][j];
                    double a=(Y-Y1)/(Y2-Y1);
                    dNdEtaPtdPtdPhi[k][i][j] = ((1.0-a)*FY1+a*FY2)*dYdEta;
                }
            }
        }
    }



} // end unnamed namespace

//void check_hadrons(const Spec & spec, std::string DataPath) {
//    int pid = 1;
//    char fname[256];
//    double scale_factor = 1000;
//
//    //#pragma omp parallel for
//    for (auto h : spec.h_HadronInfo) {
//        if (h.s[0] < 0.001) continue;
//        double mass = h.s[0];
//        double gspin = h.s[1];
//        double fermi_boson = h.s[2];
//        double muB = h.s[3];
//        std::cout << "spec_name = " << spec.particles.at(pid).name << " finished" << std::endl;
//        std::cout << "spec_monval = " << spec.particles.at(pid).monval << std::endl;
//        std::cout << "spec_mass = " << spec.particles.at(pid).mass << std::endl;
//        std::cout << h.s[0] << " " << h.s[1] << " " << h.s[2] << " " << h.s[3] << std::endl;
//        pid ++;
//    }
//} // end check_hadrons
//
//void spec_cpp(const Spec & spec, std::string DataPath) {
//    int pid = 1;
//    char fname[256];
//    double scale_factor = 1000;
//
//    //#pragma omp parallel for
//    for (auto h : spec.h_HadronInfo) {
//        if (h.s[0] < 0.001) continue;
//        std::cout << h.s[0] << " " << h.s[1] << " " << h.s[2] << " " << h.s[3] << std::endl;
//        double mass = h.s[0];
//        double gspin = h.s[1];
//        double fermi_boson = h.s[2];
//        double muB = h.s[3];
//        double dof = gspin / std::pow(2.0*M_PI, 3.0);
//        double dNdYPtdPtdPhi[NY][NPT][NPHI];
//        double dNdEtaPtdPtdPhi[NY][NPT][NPHI];
//        for( int k=0; k!=NY; k++ )
//            for( int l=0; l!=NPT; l++ )
//                for( int m=0; m!=NPHI; m++ ){
//                    dNdYPtdPtdPhi[k][l][m] = 0.0;
//                }
//
//        std::cout << "initialize finished !" << std::endl;
//
//        for (const auto & SF : spec.h_SF) {
//            double ds0 = SF.s[0];
//            double ds1 = SF.s[1];
//            double ds2 = SF.s[2];
//            double ds3 = SF.s[3];
//            double vx  = SF.s[4];
//            double vy  = SF.s[5];
//            double vh  = SF.s[6];
//            double etas  = SF.s[7];
//            double gamma = 1.0/std::sqrt(std::max(1.0E-30, 1.0-vx*vx - vy*vy - vh*vh));
//            for (int k=0; k!=NY; k++)
//                for (int l=0; l!=NPT; l++){
//                    double pt =  spec.h_Pt[l];
//                    double mt = std::sqrt(mass*mass + pt*pt);
//                    double mtcy =  mt * cosh( spec.h_Y[k] - etas );
//                    double mtsy =  mt * sinh( spec.h_Y[k] - etas );
//                    for( int m=0; m!=NPHI; m++ ){
//                        double  px = pt *  spec.h_CPhi[m];
//                        double  py = pt *  spec.h_SPhi[m];
//                        double efkt = ( gamma * ( mtcy - px*vx - py*vy - mtsy*vh) - muB ) / spec.Tfrz;
//                        double volum = mtcy*ds0 - px*ds1 - py*ds2 - mtsy*ds3;
//                        dNdYPtdPtdPhi[k][l][m] += dof * volum / ( std::exp(efkt) + fermi_boson ) * scale_factor;
//                        /** in units of GeV.fm^3 */
//                    }
//                }
//        }
//        //std::cout << '.';
//        std::cout << "cpp spec for " << spec.particles.at(pid).name << " finished" << std::endl;
//
//        int monval = spec.particles.at(pid).monval;
//        // write spec in Y, Pt, Phi
//        if( monval > 0 ) sprintf( fname, "%s/dNdYPtdPtdPhi_Reso%d.dat", DataPath.c_str(), monval );
//        else sprintf( fname, "%s/dNdYPtdPtdPhi_ResoA%d.dat", DataPath.c_str(), -monval );
//        WriteSpec(fname, dNdYPtdPtdPhi);
//
//        // write spec in Eta, Pt, Phi
//        get_dNdEtaPtdPtdPhi(spec, mass, dNdYPtdPtdPhi, dNdEtaPtdPtdPhi);
//        if( monval > 0 ) sprintf( fname, "%s/dNdEtaPtdPtdPhi_Reso%d.dat", DataPath.c_str(), monval );
//        else sprintf( fname, "%s/dNdEtaPtdPtdPhi_ResoA%d.dat", DataPath.c_str(), -monval );
//        WriteSpec(fname, dNdEtaPtdPtdPhi);
//
//        pid++;
//    }
//}

/////////////////////////////////////////////////
int main(int argc, char** argv)
{
    std::string pathin;
    int VISCOUS_ON = 0;
    int DECAY_ON = 0;
    int GPU_ID = 0;
    if (argc == 5) {
        pathin = std::string(argv[1]);
        std::string opt2(argv[2]);
        if ( opt2 == "true" || opt2 == "1" || opt2 == "True" || opt2 == "TRUE" ) {
            VISCOUS_ON = 1;
        }
        std::string opt3(argv[3]);
        if ( opt3 == "true" || opt3 == "1" || opt3 == "True" || opt3 == "TRUE" ) {
            DECAY_ON = 1;
        }
        GPU_ID = atoi(argv[4]);
    } else {
        std::cerr << "Usage: ./spec hypersf_directory viscous_on decay_on gpu_id" << std::endl;
        std::cerr << "Example: ./spec /home/name/results/event0 true true 0" << std::endl;
    }


    clock_t t;
    t = clock();
    Spec spec(pathin, VISCOUS_ON, DECAY_ON, GPU_ID);

    std::cout << "begin to calc spec" << std::endl;
    spec.CalcSpec();

    //spec_cpp(spec, pathin);
    //check_hadrons(spec, pathin);
    //spec.ReadSpec();
    
    t = clock() - t;
    std::printf("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

    if (DECAY_ON) {
        spec.ResoDecay();
    }

    spec.testResults();

    return 0;
}
