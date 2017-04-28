/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */
#include "include/particletype.h"

namespace Smash {

/** the following function only works for mesons */
void ParticleType::update_expansion_rate(double temperature) {
    double sum_prob = 0.0;
    Integrator integrate;
    for (int i=1; i<6; i++) {
      double T_expansion = temperature / double(i);
      prob[i-1] = integrate(0.0, 100.0*T_expansion,
                  [&](double p) { return
                     p*p*juttner_distribution_func(
                        p, mass, T_expansion, mu_B, 0.0);
              });
      sum_prob += prob[i-1];
    }
    for (int i=0; i<5; i++) {
      prob[i] /= sum_prob;
    }
}

double ParticleType::sample_momentum(double temperature){
    double pmag = 0.0;
    if (baryon) {
        pmag = sample_momenta1(temperature, mass);
    } else {
        double newT;
        double rand = Random::canonical();
        if (rand < prob[0]) {
            newT = temperature;
        } else if (rand < prob[0] + prob[1]) {
            newT = temperature * 0.5;
        } else if (rand < prob[0] + prob[1] + prob[2]) {
             newT = temperature * 0.33;
        } else if (rand < prob[0] + prob[1] + prob[2] + prob[3]) {
             newT = temperature * 0.25;
        } else {
             newT = temperature * 0.2;
        }
        pmag = sample_momenta1(newT, mass);
    }
}

} // namespace Smash
