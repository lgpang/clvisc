/*
 *
 *    Copyright (c) 2013-2015
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */
#include"include/sampler.h"

#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include<fstream>
#include<cmath>
#include<iostream>
#include<chrono>
#include<exception>

#include "include/angles.h"
#include "include/constants.h"
#include "include/distributions.h"
#include "include/inputfunctions.h"
#include"include/random.h"
#include "include/integrate.h"

#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))

namespace Smash {

namespace Boost{

void tensor_boost(double u[4], double Pi[][4], double PiTilde[][4]) {
    /** lorentz boost a tensor pi[mu][nu] to pi*[mu][nu] */
	int alpha,beta,gamma,delta;
	double Linv[4][4],L[4][4];  // Lorentz Boost Matrices
	double ucontra[4]={u[0],-u[1],-u[2],-u[3]},n[4]={1.0,0,0,0};
	for(alpha=0;alpha<4;alpha++){
		for(beta=0;beta<4;beta++){
			Linv[alpha][beta]=2.0*n[alpha]*ucontra[beta]-(u[alpha]+n[alpha])
                                   *(ucontra[beta]+n[beta])/(1.0+u[0]);
			if(alpha==beta) Linv[alpha][beta]+=1.0;
			L[beta][alpha]=Linv[alpha][beta];
		}
	}
	for(alpha=0;alpha<4;alpha++){
		for(delta=0;delta<4;delta++){
			PiTilde[alpha][delta]=0.0;
			for(beta=0;beta<4;beta++){
				for(gamma=0;gamma<4;gamma++){
					PiTilde[alpha][delta]+=
                         Linv[alpha][beta]*Pi[beta][gamma]*L[gamma][delta];
				}
			}
		}
	}
}
}

// constructor sampler
Sampler::Sampler(const std::string & fpath,
                 bool viscous_correction_on,
                 bool force_resonance_decay) {
    viscous_correction_ = viscous_correction_on;

    read_hypersurface(fpath);

    // if has shear viscosity, get pimn
    if ( viscous_correction_on ) {
        read_pimn_on_sf(fpath);
    }

    force_decay_ = force_resonance_decay;

    read_chemical_potential();

    read_pdg_table();

    get_equilibrium_properties();

    draw_hadron_type_.reset_weights(densities_);

    // tot density for all hadrons
    dntot_ = std::accumulate(densities_.begin(),
                             densities_.end(), 0.0);

    // get pion_after_reso / pion_before_reso ratio from equilibrium
    // density and decay chain in pdg05.dat
    double ratio = pion_ratio_after_reso_vs_before();

    std::clog << "pion after reso over before ratio=" << ratio
              << std::endl;
}


// read freeze out hyper surface
void Sampler::read_hypersurface(const std::string & fpath) {
    char buf[256];
    std::stringstream hypersf_path;
    hypersf_path << fpath << "/hypersf.dat";
    std::ifstream fin(hypersf_path.str());
    fin.getline(buf, 256);  // readin the comment
    std::string comments(buf);
    freezeout_temperature_ = std::stof(comments.substr(7));

    std::clog << "Tfrz=" << freezeout_temperature_ << std::endl;

    std::string hyper_surface = read_all(std::move(fin));

    auto hyper_surface_elements = line_parser(hyper_surface);

    double ds0, ds1, ds2, ds3, vx, vy, vetas, tau, x, y, etas;

    tau = 0.0;
    x = 0.0;
    y = 0.0;

    for ( const Line & line : hyper_surface_elements ) {
        VolumnElement vi;
        std::istringstream lineinput(line.text);

        lineinput >> ds0 >> ds1 >> ds2 >> ds3 >> vx >> vy >> vetas \
            >> etas;
        //    >> tau >> x >> y >> etas;
        if ( lineinput.fail() ) {
            throw LoadFailure(build_error_string(
                        "While loading freeze out hypersurface:\n"
                        "Failed to convert the input string to the "
                        "expected volume element.", line));
        } else {
            vi.dsigma = FourVector(ds0, ds1, ds2, ds3);

            vi.velocity = ThreeVector(vx, vy, vetas);

            vi.position = FourVector(tau, x, y, etas);

            elements_.emplace_back(std::move(vi));
        }
    }

    std::clog << "num of sf = " << elements_.size() << std::endl;
}

// read pimn on the freeze out hyper surface
void Sampler::read_pimn_on_sf(const std::string & fpath) {
    std::stringstream pimn_path;
    pimn_path << fpath << "/pimnsf.dat";
    std::ifstream fin(pimn_path.str());
    char buf[256];
    fin.getline(buf, 256);  // readin the comment
    std::string comments(buf);
    one_over_2TsqrEplusP_ = std::stof(comments.substr(20));

    std::clog << "one_over_2TsqrEplusP_=" << one_over_2TsqrEplusP_
              << std::endl;
    std::string pimn_on_sf = read_all(std::move(fin));

    auto pimns = line_parser(pimn_on_sf);

    unsigned long number_of_elements = pimns.size();

    if ( number_of_elements != elements_.size() ) {
        std::cerr << "hypersf.dat and pimn.dat have different size,";
        std::cerr << "switch back to ideal hydro" << std::endl;
        viscous_correction_ = false;
    } else {
        double pimn[4][4];
        // pimn_star: pimn in comoving frame of fluid cell
        double pimn_star[4][4];
        double umu[4];
        for ( unsigned long n = 0; n < number_of_elements; n++ ) {
            auto line = pimns[n];
            std::istringstream lineinput(line.text);
            for ( int i=0; i < 4; i++ ) {
                for ( int j=i; j<4; j++ ) {
                    lineinput >> pimn[i][j];
                    if ( i != j ) pimn[j][i] = pimn[i][j];
                }
            }

            if ( lineinput.fail() ) {
                throw LoadFailure(build_error_string(
                            "While loading pi^{mu nu} on freeze out hypersurface:\n"
                            "Failed to convert the input string to the "
                            "expected 10 pimn tensors.", line));
            } else {
                VolumnElement * ele = &elements_[n];
                double utau;

                double vsqr = ele->velocity*ele->velocity;
                if ( vsqr < 1.0 ) {
                    utau = 1.0/std::sqrt(1.0-vsqr);
                    umu[0] = utau;
                    umu[1] = utau * ele->velocity.x1();
                    umu[2] = utau * ele->velocity.x2();
                    umu[3] = utau * ele->velocity.x3();
                } else {
                    umu[0] = 1.0;
                    umu[1] = 0.0;
                    umu[2] = 0.0;
                    umu[3] = 0.0;
                }

                // boost tensor pimn to pimn_star
                Boost::tensor_boost(umu, pimn, pimn_star);

                double pimn_max = 0.0;
                for ( int i=0; i < 4; i++ ) {
                    for ( int j=i; j<4; j++ ) {
                        double pi_ij = pimn_star[i][j];
                        ele->pimn[idx(i, j)] = pi_ij;
                        pimn_max += pi_ij*pi_ij;
                        if ( i != j ) pimn_max += pi_ij*pi_ij;
                    }
                }

                pimn_max = std::sqrt(pimn_max);

                ele->pimn_max = pimn_max;
            } // end else
        } // end loop for all elements
    }
}

void Sampler::read_pdg_table() {
    std::ifstream fin("pdg05.dat");

    if( fin.is_open() ){
        while(fin.good()){
            ParticleType p;
            p.stable = 0;
            fin>>p.pdgcode>>p.name>>p.mass>>p.width         \
                >>p.gspin>>p.baryon>>p.strange>>p.charm     \
                >>p.bottom>>p.gisospin>>p.charge>>p.decays;

            p.antibaryon_spec_exists = 0;

            if ( fin.eof() ) break;
            DecayChannel dec;

            std::vector<double> branch_ratios;

            // store branch_ratios which is used to initialize one
            // discrete distribution function for each resonance

            for(int k=0; k<p.decays; k++){
                fin >> dec.pidR >> dec.num_of_daughters
                    >> dec.branch_ratio
                    >> dec.daughters[0] >> dec.daughters[1]
                    >> dec.daughters[2] >> dec.daughters[3]
                    >> dec.daughters[4];

                dec.num_of_daughters = std::abs(dec.num_of_daughters);
                // don't count if decay to itself
                if ( dec.num_of_daughters != 1 ) {
                    p.decay_channels.push_back(dec);
                    branch_ratios.push_back(dec.branch_ratio);
                }
            }

            p.mu_B = muB_[p.pdgcode];

            p.draw_decay = Random::discrete_dist<double>(branch_ratios);

            if(p.width < 1.0E-8)p.stable=1;
            list_hadrons_.push_back(p);
        }
        fin.close();
    } else {
        std::cerr<<"#Failed to open pdg data table\n";
        exit(0);
    }

    // update newpid which is used in the antiB
    unsigned long N=list_hadrons_.size();
    for ( unsigned long i =0; i< N; i++ ) {
        newpid[ list_hadrons_[i].pdgcode ] =  i;
    }


    for(std::vector<ParticleType>::size_type i=0; i!=N; i++){
        if ( list_hadrons_[i].baryon ) {
            ParticleType antiB;//anti-baryon
            antiB.pdgcode = -list_hadrons_[i].pdgcode;
            antiB.name   = "A";
            antiB.name.append(list_hadrons_[i].name);
            antiB.mass = list_hadrons_[i].mass;
            antiB.width = list_hadrons_[i].width;
            antiB.gspin = list_hadrons_[i].gspin;
            antiB.baryon = -list_hadrons_[i].baryon;
            antiB.strange= -list_hadrons_[i].strange;
            antiB.charm = -list_hadrons_[i].charm;
            antiB.bottom = -list_hadrons_[i].bottom;
            antiB.gisospin=list_hadrons_[i].gisospin;
            antiB.charge=-list_hadrons_[i].charge;
            antiB.decays=list_hadrons_[i].decays;
            antiB.stable=list_hadrons_[i].stable;
            antiB.antibaryon_spec_exists = 1;
            antiB.draw_decay = list_hadrons_[i].draw_decay;
            antiB.mu_B = list_hadrons_[i].mu_B;

            for ( const auto & channel : list_hadrons_[i].decay_channels) {
                DecayChannel anti_baryon_decay;
                anti_baryon_decay.pidR = -channel.pidR;
                anti_baryon_decay.num_of_daughters = channel.num_of_daughters;
                anti_baryon_decay.branch_ratio = channel.branch_ratio;
                for ( int j = 0; j < channel.num_of_daughters; j++ ) {
                    int nid = newpid[channel.daughters[j]];
                    ParticleType *daughter = &list_hadrons_[nid];
                    //std::clog << "daughter->charge=" << daughter->charge << std::endl;
                    if ( daughter->charge != 0 ) {
                        anti_baryon_decay.daughters[j] = -daughter->pdgcode;
                    } else {
                        if ( daughter->baryon ) {
                            // for neutral baryons; like anti-neutron
                            anti_baryon_decay.daughters[j] = -daughter->pdgcode;
                        } else {
                            // for neutral mesons 
                            anti_baryon_decay.daughters[j] = daughter->pdgcode;
                        }
                    }
                }
                antiB.decay_channels.push_back(anti_baryon_decay);
            }

            list_hadrons_.push_back(antiB);
        }
    }

    unsigned long SizePID = list_hadrons_.size();

    for( unsigned long i = N; i < SizePID; i++ ){
        newpid[ list_hadrons_[i].pdgcode ] =  i;
    }

    std::clog<<"newpid of pion = "<<newpid[ 211 ]<<std::endl;
    std::clog<<"newpid of proton = "<<newpid[ 2212 ]<<std::endl;
    std::clog<<"newpid of -13334 = "<<newpid[ -13334 ]<<std::endl;

}

void Sampler::read_chemical_potential() {
    int pid;
    double chemicalpotential;
    std::ifstream fin("chemical_potential.dat");
    if( fin.is_open() ){
        while( fin.good() ){
            fin>>pid>>chemicalpotential;
            if( fin.eof() )break;  // eof() repeat the last line
            muB_[ pid ] = chemicalpotential;
            std::clog<<"pid="<<pid<<" muB="<<muB_[pid]<<std::endl;
        }
        fin.close();
    } else {
        std::cerr<<"#Can't open muB data file!\n";
        exit(0);
    }
}



    /**unnamed namespace that stores helper datatypes that will only
     * used in the equilibrium density calculation*/
namespace {
    /// prefactor == 4*pi/(twopi*hbarc)**3.0
    constexpr double prefactor = 1.0/(2.0*M_PI*M_PI*hbarc*hbarc*hbarc);

    Integrator integrate;

    double get_equilibrium_density(const double mass, \
            const double temperature, \
            const double baryon_chemical_potential, \
            const double fermion_boson_factor, \
            const double spin_degeneracy) {
        // Integrate over |p| to get d^3p/(2pi)^3 f(p)


        double integral_value = integrate(0.0, 100.0*temperature,
                [&](double momentum_radius) { return
                   momentum_radius*momentum_radius*juttner_distribution_func(
                      momentum_radius, mass,
                      temperature, baryon_chemical_potential,
                      fermion_boson_factor);
            });

        return spin_degeneracy*prefactor*integral_value;
    }
}  // end unnamed namespace


    // Calc equilibrium density, initialize adaptive rejection sampler
    // to sample the momentum in local rest frame for each hadron
    void Sampler::get_equilibrium_properties() {
        densities_.resize(0);
        particle_dist_.resize(0);
        for ( const auto & particle_type : list_hadrons_ ) {
            double baryon_chemical_potential = particle_type.mu_B;
            //double baryon_chemical_potential = 0.0;
            // determine fermi-dirac or bose-einstein distribution
            double baryon_meson_factor = -1.0;
            if ( particle_type.baryon != 0 ) baryon_meson_factor = 1.0;

            // calc the equilibrium density
            double density = get_equilibrium_density(particle_type.mass,
                    freezeout_temperature_, baryon_chemical_potential,
                    baryon_meson_factor, particle_type.gspin);

            // set density = 0 for photon to turn off sampling
            if ( particle_type.pdgcode == 22 ) density = 0.0;

            densities_.push_back(density);

            // initialize the adaptive rejection sampler for each species
            Rejection::AdaptiveRejectionSampler juttner([&](double x) {
                    return x*x*juttner_distribution_func(x, particle_type.mass, \
                        freezeout_temperature_, baryon_chemical_potential, \
                        baryon_meson_factor); \
                    }, 0.0, 15.0);

            particle_dist_.emplace_back(std::move(juttner));
        }
    }

    /** (1) Get dntot for one small piece of freeze out hyper surface, and us it
     *  as the mean value of Poisson distribution to sample number of hadrons Ni
     *  from this freeze out hyper surface;
     *  (2) Use partial density dni/dntot as probability for discrete 
     *  distribution  to determine the particle type
     *  (3) Sample momentum for sampled particle type in local rest frame,
     *  keep or reject according to weight function
     *      \f$ p^{\mu} d\Sigma_{\mu} / p^{0} \f$
     *  (5)boost it with fluid velocity
     *
     *  The difficulties:
     *  (1) Sample momentum from boson or fermion distribution with finite 
     *  chemical potential (use Adaptive Rejection Sampler).
     *  (2) There are two ways to calculate dntot and do rejection
     *      A: \f$ dntot = density * (u_{\mu} d\Sigma^{\mu}) \f$
     *      fix total number, do repeating if 
     *      \f$ rand > p^{\mu*} d\Sigma_{\mu*} / p^{0*} \f$.
     *
     *      B: \f$ dntot = density * sigma_max \f$
     *      reject if \f$rand > p^{\mu*} d\Sigma_{\mu*}/(p^{0*}*sigma_max)\f$
     *
     *      However, the second method gives 1-2% more particles than the first
     *      method. Why?
     *      I guess there are some freeze out hyper surface whose 
     *      \f$ u_{\mu} d\Sigma^{\mu} ==0 \f$, and in method A, dntot=0 is set
     *      in the begining.
     */

    void Sampler::sample_particles_from_hypersf() {

        Angles direction;
        FourVector momentum_in_lrf;
        FourVector momentum;
        FourVector position;

        FourVector sigma_lrf;

        for ( auto & ele : elements_ ) {

            /** Boost dSigma to local rest frame */
            sigma_lrf = ele.dsigma.LorentzBoost(ele.velocity);

            double udotsigma = sigma_lrf.x0();

            if ( udotsigma < 0.0 ) continue;

            double sigmamax = udotsigma + std::sqrt(sigma_lrf.sqr3());

            double dNtot = dntot_ * udotsigma;
            //double dNtot = dntot_*sigmamax;

            int Ni = Random::poisson(dNtot);

            double cosheta, sinheta;

            /* \todo: Although the possibility is quite small, it is possilbe for
             * 2 particles to have the same position (if Ni>1) 
             * Scattering may happen after produced.*/
            if ( Ni != 0 ) {
                cosheta = cosh(ele.position[3]);
                sinheta = sinh(ele.position[3]);
                position.set_x0(ele.position[0]*cosheta);
                position.set_x1(ele.position[1]);
                position.set_x2(ele.position[2]);
                position.set_x3(ele.position[0]*sinheta);
            }

            /** pmag = |\vec{momentum}| */
            double pmag;

            /* nid: Determine the particle type from discrete distribution*/
            for ( int sampled_hadrons = 0; sampled_hadrons < Ni; sampled_hadrons++ ) {
                int nid = draw_hadron_type_();

                double mass = list_hadrons_.at(nid).mass;

                //pmag = particle_dist_.at(nid).get_one_sample();
                pmag = sample_momenta1(freezeout_temperature_, mass);

                int while_loop_num = 0;

                while ( true ) {
                    direction.distribute_isotropically();

                    /// momentum in local rest frame
                    momentum_in_lrf = FourVector(std::sqrt(pmag*pmag+mass*mass),
                            pmag*direction.threevec());

                    double pdotsigma = momentum_in_lrf.Dot(sigma_lrf);

                    double p0_star = momentum_in_lrf[0];
                    double weight_visc = pdotsigma/(p0_star*sigmamax);

                    if ( viscous_correction_ ) {
                        double pmu_pnu_pimn = momentum_in_lrf[0]*momentum_in_lrf[0]*ele.pimn[0]
                                      + momentum_in_lrf[1]*momentum_in_lrf[1]*ele.pimn[4]
                                      + momentum_in_lrf[2]*momentum_in_lrf[2]*ele.pimn[7]
                                      + momentum_in_lrf[3]*momentum_in_lrf[3]*ele.pimn[9]
                                      - 2.0*momentum_in_lrf[0]*momentum_in_lrf[1]*ele.pimn[1]
                                      - 2.0*momentum_in_lrf[0]*momentum_in_lrf[2]*ele.pimn[2]
                                      - 2.0*momentum_in_lrf[0]*momentum_in_lrf[3]*ele.pimn[3]
                                      + 2.0*momentum_in_lrf[1]*momentum_in_lrf[2]*ele.pimn[5]
                                      + 2.0*momentum_in_lrf[1]*momentum_in_lrf[3]*ele.pimn[6]
                                      + 2.0*momentum_in_lrf[2]*momentum_in_lrf[3]*ele.pimn[8];

                        double f0;   // equilibrium distribution
                        double baryon_chemical_potential = list_hadrons_.at(nid).mu_B;
                        double fermion_boson_factor;

                        // alpha is the regulation factor from ShenChun
                        double alpha = 1.9;
                        //double regulation = std::pow(pmag/freezeout_temperature_, alpha - 2.0);
                        double regulation = 1.0;

                        if ( list_hadrons_.at(nid).baryon ) {
                            fermion_boson_factor = 1.0;
                            f0 = juttner_distribution_func(pmag, mass, \
                                        freezeout_temperature_, baryon_chemical_potential,
                                        fermion_boson_factor);

                            if ( f0 > 1.0 ) {
                                std::clog << "(F) After f0=" << f0 << "  > 1.0" << std::endl;
                            }

                            weight_visc *= (1.0 + regulation * (1.0 - f0)*pmu_pnu_pimn*one_over_2TsqrEplusP_);
                            weight_visc /= (1.0+regulation * p0_star*p0_star*ele.pimn_max*one_over_2TsqrEplusP_);

                        } else {
                            fermion_boson_factor = -1.0;
                            f0 = juttner_distribution_func(pmag, mass, \
                                        freezeout_temperature_, baryon_chemical_potential,
                                        fermion_boson_factor);

                            if ( f0 > 1.1 ) {
                                std::clog << "(B) f0=" << f0 << "  > 1.0" << std::endl;
                            }
                            weight_visc *= (1.0 + regulation * (1.0 + f0)*pmu_pnu_pimn*one_over_2TsqrEplusP_);
                            weight_visc /= (1.0+ regulation *2.1*p0_star*p0_star*
                                    ele.pimn_max*one_over_2TsqrEplusP_);
                        }
                    }

                    while_loop_num ++;
                    if ( while_loop_num > 10000 &&  weight_visc <= 0.0 ) {
                      std::clog << "more than 10000 loops for this cell, skip it ..." << std::endl;
                      std::clog << "smaller than 0 weight_visc=" << weight_visc << std::endl;
                      std::clog << "pdotsigma=" << pdotsigma << std::endl;
                      std::clog << "p0_star=" << p0_star << std::endl;
                      std::clog << "sigmamax=" << sigmamax << std::endl;
                      std::clog << "momentum_lrf=" << momentum_in_lrf << std::endl;
                      std::clog << "sigma_lrf=" << sigma_lrf << std::endl;
                      std::clog << "mass = " << mass << std::endl;
                      std::clog << std::endl;
                      break;
                    }


                    /// keep or reject 
                    if ( Random::canonical() < weight_visc ) {
                            ParticleData particle; 
                            particle.pdgcode = list_hadrons_.at(nid).pdgcode;
                            particle.position = position;

                            momentum = momentum_in_lrf.LorentzBoost(-ele.velocity);

                            double mt = std::sqrt(momentum[0]*momentum[0]-
                                    momentum[3]*momentum[3]);

                            double Y = std::atanh(momentum[3]/momentum[0])
                                    + ele.position[3];

                            momentum = FourVector(mt*std::cosh(Y), momentum.x1(),
                                            momentum.x2(), mt*std::sinh(Y));

                            particle.momentum = momentum;

                            // if stable and charged, store
                            if ( list_hadrons_.at(nid).stable == 1 ) {
                                particles_.push_back(particle);
                            } else {
                                if ( force_decay_ ) {
                                    force_decay(particle);
                                } else {
                                    particles_.push_back(particle);
                                }
                            }

                            // if one hadron is sampled, break while true loop
                            break;
                    }
                } // end while(true)
            }
        }
    }



    void Sampler::force_decay(const ParticleData & resonance) {
        ParticleData tmp = resonance;
        std::vector<ParticleData> unstable_daughters;
        // initialize with the resonance
        unstable_daughters.push_back(tmp);
        // do resonance decay while any of the daughters are unstable
        do {
            ParticleData reso_data = unstable_daughters.back();

            unstable_daughters.pop_back();

            int newid = newpid[reso_data.pdgcode];
            ParticleType reso_type = list_hadrons_.at(newid);


            int channel_id = reso_type.draw_decay();

            DecayChannel channel = reso_type.decay_channels.at(channel_id);

            // std::clog << (channel.pidR == reso_type.pdgcode) << std::endl;
            int n = channel.num_of_daughters;
            if ( n == 2 ) {
                two_body_decay(reso_data, channel, unstable_daughters);
            } else if ( n == 3 ) {
                three_body_decay(reso_data, channel, unstable_daughters);
            }

        } while (!unstable_daughters.empty());
    }

    // help function to get four-momentum from mass and three_momentum
    FourVector four_momentum(double mass, ThreeVector three_momentum) {
        return FourVector(std::sqrt(mass*mass + three_momentum*three_momentum),
                three_momentum);
    }

    void Sampler::two_body_decay(const ParticleData & reso_data,
                        const DecayChannel & channel,
                        std::vector<ParticleData> & unstable) {
        int id_R = newpid[channel.pidR];
        int id_a = newpid[channel.daughters[0]];
        int id_b = newpid[channel.daughters[1]];

        ParticleType *resonance = &list_hadrons_[id_R];
        ParticleType *outgoing_a = &list_hadrons_[id_a];
        ParticleType *outgoing_b = &list_hadrons_[id_b];

        double mR = resonance->mass;
        double ma = outgoing_a->mass;
        double mb = outgoing_b->mass;

        //if ( ma + mb > mR ) {
        //    std::clog << "pidR" << channel.pidR << std::endl;
        //    std::clog << "pid_a=" << channel.daughters[0] << std::endl;
        //    std::clog << "pid_b=" << channel.daughters[1] << std::endl;
        //    std::clog << ma << "+" << mb << ">" << mR << std::endl;
        //}
        
        // notice: add this in the code drive some 3-body decay channels
        // violate the total energy conservation a little bit
        while ( ma + mb > mR ) {
                mR += 0.25 * resonance->width;
                ma -= 0.5  * outgoing_a->width;
                mb -= 0.5  * outgoing_b->width;
        }

        // pstar = |p0*| = |p1*| in local rest frame
        double pstar = std::sqrt((mR*mR-(ma+mb)*(ma+mb)) *
                                 (mR*mR-(ma-mb)*(ma-mb)))/(2.0*mR);

        ThreeVector velocity_of_reso = reso_data.momentum.velocity();

        Angles direction;

        direction.distribute_isotropically();

        FourVector momentum_a = four_momentum(ma, pstar*direction.threevec());
        FourVector momentum_b = four_momentum(mb, -momentum_a.threevec());

        momentum_a = momentum_a.LorentzBoost(-velocity_of_reso);
        momentum_b = momentum_b.LorentzBoost(-velocity_of_reso);

        if ( outgoing_a->pdgcode != 22 ) {
            ParticleData a_data;
            a_data.pdgcode = outgoing_a->pdgcode;
            a_data.position = FourVector(0.0, 0.0, 0.0, 0.0);
            a_data.momentum = momentum_a;

            if ( outgoing_a->stable ) {
                particles_.push_back(a_data);
            } else {
                unstable.push_back(a_data);
            }
        }

        // for the second particle
        if ( outgoing_b->pdgcode != 22 ) {
            ParticleData b_data;
            b_data.pdgcode = outgoing_b->pdgcode;
            b_data.position = FourVector(0.0, 0.0, 0.0, 0.0);
            b_data.momentum = momentum_b;

            if ( outgoing_b->stable ) {
                particles_.push_back(b_data);
            } else {
                unstable.push_back(b_data);
            }
        }
    }

    void Sampler::three_body_decay(const ParticleData & reso_data,
                        const DecayChannel & channel,
                        std::vector<ParticleData> & unstable) {
        int id_R = newpid[channel.pidR];
        int id_a = newpid[channel.daughters[0]];
        int id_b = newpid[channel.daughters[1]];
        int id_c = newpid[channel.daughters[2]];

        ParticleType *resonance = &list_hadrons_[id_R];
        ParticleType *outgoing_a = &list_hadrons_[id_a];
        ParticleType *outgoing_b = &list_hadrons_[id_b];
        ParticleType *outgoing_c = &list_hadrons_[id_c];

        double mass_R = resonance->mass;
        double mass_a = outgoing_a->mass;
        double mass_b = outgoing_b->mass;
        double mass_c = outgoing_c->mass;

        double s_ab_max = (mass_R - mass_c)*(mass_R - mass_c);
        double s_ab_min = (mass_a + mass_b)*(mass_a + mass_b);
        double s_bc_max = (mass_R - mass_a)*(mass_R - mass_a);
        double s_bc_min = (mass_b + mass_c)*(mass_b + mass_c);

        /* randomly pick values for s_ab and s_bc
         * until the pair is within the Dalitz plot */
        double dalitz_bc_max = 0.0, dalitz_bc_min = 1.0;
        double s_ab = 0.0, s_bc = 0.5;
        while (s_bc > dalitz_bc_max || s_bc < dalitz_bc_min) {
          s_ab = Random::uniform(s_ab_min, s_ab_max);
          s_bc = Random::uniform(s_bc_min, s_bc_max);
          const double e_b_rest =
            (s_ab - mass_a * mass_a + mass_b * mass_b) / (2 * std::sqrt(s_ab));
          const double e_c_rest =
            (mass_R * mass_R - s_ab - mass_c * mass_c) /
            (2 * std::sqrt(s_ab));
          dalitz_bc_max = (e_b_rest + e_c_rest) * (e_b_rest + e_c_rest) -
                          (std::sqrt(e_b_rest * e_b_rest - mass_b * mass_b) -
                           std::sqrt(e_c_rest * e_c_rest - mass_c * mass_c)) *
                           (std::sqrt(e_b_rest * e_b_rest - mass_b * mass_b) -
                            std::sqrt(e_c_rest * e_c_rest - mass_c * mass_c));
          dalitz_bc_min = (e_b_rest + e_c_rest) * (e_b_rest + e_c_rest) -
                          (std::sqrt(e_b_rest * e_b_rest - mass_b * mass_b) +
                           std::sqrt(e_c_rest * e_c_rest - mass_c * mass_c)) *
                           (std::sqrt(e_b_rest * e_b_rest - mass_b * mass_b) +
                            std::sqrt(e_c_rest * e_c_rest - mass_c * mass_c));
        }

        const double energy_a =
            (mass_R * mass_R + mass_a * mass_a - s_bc) / (2 * mass_R);
        const double energy_c =
            (mass_R * mass_R + mass_c * mass_c - s_ab) / (2 * mass_R);
        const double energy_b =
            (s_ab + s_bc - mass_a * mass_a - mass_c * mass_c) / (2 * mass_R);
        const double momentum_a = std::sqrt(energy_a * energy_a - mass_a * mass_a);
        const double momentum_c = std::sqrt(energy_c * energy_c - mass_c * mass_c);
        const double momentum_b = std::sqrt(energy_b * energy_b - mass_b * mass_b);

        // const double total_energy = reso_data.momentum.abs();

        /* momentum_a direction is random */
        Angles phitheta;

        ThreeVector velocity_of_reso = reso_data.momentum.velocity();

        ParticleData a_data, b_data, c_data;

        phitheta.distribute_isotropically();
        /* This is the angle of the plane of the three decay particles */
        a_data.momentum = four_momentum(mass_a, phitheta.threevec() * momentum_a);
      
        /* Angle between a and b */
        double theta_ab = std::acos(
            (energy_a * energy_b - 0.5 * (s_ab - mass_a * mass_a - mass_b * mass_b)) /
            (momentum_a * momentum_b));

        bool phi_has_changed = phitheta.add_to_theta(theta_ab);
        b_data.momentum = four_momentum(mass_b, phitheta.threevec() * momentum_b);
      
        /* Angle between b and c */
        double theta_bc = std::acos(
            (energy_b * energy_c - 0.5 * (s_bc - mass_b * mass_b - mass_c * mass_c)) /
            (momentum_b * momentum_c));
        // pass information on whether phi has changed during the last adding
        // on to add_to_theta:
        phitheta.add_to_theta(theta_bc, phi_has_changed);

        c_data.momentum = four_momentum(mass_c, phitheta.threevec() * momentum_c);
        
        /* Momentum check : why (mR, ma, mb) correction breaks 
         * the energy conservation here */

        // FourVector ptot = a_data.momentum + b_data.momentum +
        //                   c_data.momentum;
        
        // if (std::abs(ptot.x0() - total_energy) > really_small) {
        //     std::clog << "1->3 energy not conserved! Before: ";
        //     std::clog << total_energy << " After: " << ptot.x0();
        //     std::clog << std::endl;
        // }

        a_data.pdgcode = outgoing_a->pdgcode;
        b_data.pdgcode = outgoing_b->pdgcode;
        c_data.pdgcode = outgoing_c->pdgcode;

        a_data.momentum = a_data.momentum.LorentzBoost(-velocity_of_reso);
        b_data.momentum = b_data.momentum.LorentzBoost(-velocity_of_reso);
        c_data.momentum = c_data.momentum.LorentzBoost(-velocity_of_reso);

        if ( outgoing_a->pdgcode != 22 ) {
            if ( outgoing_a->stable ) {
                particles_.push_back(a_data);
            } else {
                unstable.push_back(a_data);
            }
        }

        if ( outgoing_b->pdgcode != 22 ) {
            if ( outgoing_b->stable ) {
                particles_.push_back(b_data);
            } else {
                unstable.push_back(b_data);
            }
        }

        if ( outgoing_c->pdgcode != 22 ) {
            if ( outgoing_c->stable ) {
                particles_.push_back(c_data);
            } else {
                unstable.push_back(c_data);
            }
        }

    }


    typedef struct {
        int type_id;
        double density; // equilibrium density
    } Resonance;

    /** Get the pion+ yield ratio between after reso ( calculated from
     * thermal equilbirum densities and branch ratios to pion+ ) and 
     * before reso */
    double Sampler::pion_ratio_after_reso_vs_before(){
        double density_pion_before_reso = densities_[1];
        double density_pion_after_reso = densities_[1];
        int i = 0;
        for ( const auto & particle_type : list_hadrons_ ) {
            std::vector<Resonance> resos;
            Resonance reso_tmp;
            reso_tmp.type_id = newpid[particle_type.pdgcode];
            reso_tmp.density = densities_[i];
            resos.push_back(reso_tmp);
            
            do {
                Resonance reso = resos.back();
                resos.pop_back();

                ParticleType reso_type = list_hadrons_[reso.type_id];

                for ( const auto & channel : reso_type.decay_channels ) {
                    for ( int nd = 0; nd < channel.num_of_daughters; nd ++ ){
                        double density_daughter = reso.density * channel.branch_ratio;
                        int new_daughter_id = channel.daughters[nd];
                        if ( new_daughter_id == 211 ) {
                            density_pion_after_reso += density_daughter;
                        } else {
                            if ( ! list_hadrons_[newpid[new_daughter_id]].stable ) {
                                reso_tmp.type_id = newpid[new_daughter_id];
                                reso_tmp.density = density_daughter;
                                resos.push_back(reso_tmp);
                                // decay those unstable particles to pion+
                            }
                        }
                    }
                }
            } while (!resos.empty());

            std::clog << "density[" << particle_type.pdgcode
                      << "]=" << densities_[i] << std::endl;

            i++;
        }

        return density_pion_after_reso / density_pion_before_reso;
    }



}  // end namespace Smash

