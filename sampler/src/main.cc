#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <iostream>

#include "include/sampler.h"
#include "include/constants.h"
#include "include/fourvector.h"
#include "include/integrate.h"

using namespace Smash;


/** Calc total number of species i from freeze out hypersurface in T.hirano's
 *  mthod. This equals to \f[ n*\sum_i u_i\cdot d\Sigma_i \f]*/
namespace hirano_method {
    constexpr double acu = 1.0E-8;
    /// dN/dp3
    inline double f(double p, double dS0, double dSi,
            double mass, double Tfrz, double muB, double lam){
        double E = sqrt(p*p+mass*mass);
        double f = 0.0;

        if( dS0 > 0.0 ){
            f += 4.0*M_PI*p*p/( exp((E-muB)/Tfrz)
                    + lam ) * dS0;
        }

        double vsig = dS0 / fmax(acu,dSi);

        //The second part in Ntotal calculation
        if ( fabs(vsig) < 1.0 && fabs(dSi)>acu && p > (mass*fabs(vsig)/std::sqrt(1-vsig*vsig)) ) {
            f += M_PI*( dSi*p*(E*E*vsig*vsig + p*p)/E/( exp((E-muB)/Tfrz) + lam) \
                    - 2.0*fabs( dS0 )*p*p/( exp((E-muB)/Tfrz) + lam) );
        }
        return f;
    }

    const double prefactor = 1.0/pow(twopi*hbarc,3.0);

    /// return: dN total number of hadrons from freeze out hyper surface dS^{mu}
    inline double get(double dS0, double dSi, double mass, double Tfrz, double muB, double lam){
        //Integrate over |p| to get dN
        Integrator integrate;

        return prefactor * integrate(0.0, 50.0*Tfrz,
                [&](double x){ return f(x, dS0, dSi, mass, Tfrz, muB, lam);});
    }
} // end namespace hirano_method




int main(int argc, char ** argv) {
    if ( argc != 4 ) {
        std::cout << "usage:" << std::endl;
        std::cout << "./main hypersf_directory viscous_on_" << std::endl;
        std::cout << "hypersf_directory: directory that has";
        std::cout << "hypersf.dat and pimnsf.dat" << std::endl;
        std::cout << "viscous_on: true to use viscous corrections" << std::endl;
        std::cout << "force_decay: true to force decay" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string path(argv[1]);

    bool viscous_on = false;
    std::string opt2(argv[2]);
    if ( opt2 == "true" || opt2 == "1" || opt2 == "True" || opt2 == "TRUE" ) {
        viscous_on = true;
    }
    // switch for force resonance decay
    bool force_decay = false;
    std::string opt3(argv[3]);
    if ( opt3 == "true" || opt3 == "1" || opt3 == "True" || opt3 == "TRUE" ) {
        force_decay = true;
    }

    Sampler sampler(path, viscous_on, force_decay);

    std::cout << "initialize finished!" << std::endl;

    constexpr int number_of_events = 1000;

    for ( int nevent=0; nevent < number_of_events; nevent++ ) {
        sampler.sample_particles_from_hypersf();
        std::cout << "event " << nevent << " finished\n";
    }

    int num_of_pion_plus = 0;
    std::stringstream fname_particle_list;
    fname_particle_list << path << "/mc_particle_list.dat";
    std::ofstream fpmag(fname_particle_list.str());

    fpmag << "#E px py pz Y pid\n";
    for ( const auto & par : sampler.particles_ ) {
        int nid = sampler.newpid[par.pdgcode];
        if ( sampler.list_hadrons_.at(nid).stable &&
             sampler.list_hadrons_.at(nid).charge ) {
        FourVector momentum = par.momentum;
        fpmag << momentum.x0() << ' ' << momentum.x1() << ' '
            << momentum.x2() << ' ' << momentum.x3() << ' '
            << 0.5*std::log((momentum.x0()+momentum.x3())/
                    (momentum.x0()-momentum.x3())) << ' '
            << par.pdgcode << std::endl;
        }
        if ( nid == 1 ) num_of_pion_plus ++;
    }
    fpmag.close();

    std::cout << "ntot for pion+ from sample=" << num_of_pion_plus/ \
        static_cast<float>(number_of_events)  << std::endl;

    double pion_mass = 0.13957;
    double baryon_chemical_potential = 0.0;
    double fermion_boson_factor = -1.0;
    double freezeout_temperature = sampler.freezeout_temperature_;

    double pionplus_density = sampler.densities_.at(1);

    double ntotal_pion_plus_from_nudotsigma = 0.0;
    double ntotal_pion_plus_from_hirano = 0.0;

    for ( const auto & ele : sampler.elements_ ) {
        FourVector sigma_lrf = ele.dsigma.LorentzBoost(ele.velocity);
        double dSi = std::sqrt(fmax(really_small*really_small, \
                    sigma_lrf.sqr()));

        ntotal_pion_plus_from_nudotsigma += pionplus_density*sigma_lrf.x0();

        ntotal_pion_plus_from_hirano += hirano_method::get(sigma_lrf.x0(), dSi, \
                pion_mass, freezeout_temperature, baryon_chemical_potential,\
                fermion_boson_factor);
    }

    std::cout << "ntot for pion+ from udotsigma=" << 
        ntotal_pion_plus_from_nudotsigma << std::endl;

    std::cout << "ntot for pion+ from hirano=" << 
        ntotal_pion_plus_from_hirano << std::endl;

}

