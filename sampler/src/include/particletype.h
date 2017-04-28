/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#ifndef SRC_INCLUDE_PARTICLETYPE_H_
#define SRC_INCLUDE_PARTICLETYPE_H_

#include <cmath>
#include <vector>
#include <deque>
#include <functional>
#include <map>
#include "distributions.h"
#include "integrate.h"
#include "random.h"

namespace Smash {

typedef struct {
    int  pidR;
    int  num_of_daughters;
    int  daughters[5];
    double branch_ratio;
}DecayChannel;


class ParticleType {
  public:
    int pdgcode;
    std::string name;
    double mass;
    double width;
    double gspin;
    int baryon;
    int strange;
    int charm;
    int bottom;
    int gisospin;
    int charge;
    int decays;
    int stable;
    int  antibaryon_spec_exists;
    // baryon chemical potential
    double mu_B;
    std::vector<DecayChannel> decay_channels;
    // discrete distribution to draw decay channel
    Random::discrete_dist<double> draw_decay;
    /** expanded probabilities for bosons
     * Sn = a + a r + a r*r + a r*r*r + ...
     *    ~ a / (1 - r) 
     * f = 1 / (e^{pu/T} -1) = e^{-pu/T} /( 1 - e^{-pu/T} )
     * since r = e^{pu/T} < 1.0, the upper equation can be
     * expanded as: f(r) = r + r*r + r^3 + r^4 + r^5 
     * the $int dp p^2 f(p)$ thus has 5 terms
     * whose discrete probabilites are thermal distribution with
     * temperature T, T/2, T/3, T/4, T/5
     * for fermions, the lightest fermion has mass~1 GeV,
     * exp(pu/T) > exp(m/T) >> 1 and 1 can be neglected.
     */
    double prob[5];

    ParticleType() = default;

    /** the following function only works for mesons */
    void update_expansion_rate(double temperature); 

    /** sample momentum magnitude for baryons and mesons*/
    double sample_momentum(double temperature);
};

} // namespace Smash

#endif // SRC_INCLUDE_PARTICLETYPE_H_
