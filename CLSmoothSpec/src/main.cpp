#include<cl_spec.h>
#include<ctime>
#include<cstdlib>

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

    Spec spec(pathin, VISCOUS_ON, DECAY_ON, GPU_ID);
 
    std::cout << "begin to calc spec" << std::endl;
    spec.CalcSpec();

    //spec.ReadSpec();

    if (DECAY_ON) {
      spec.ResoDecay();
    }

    spec.testResults();

    return 0;
}
