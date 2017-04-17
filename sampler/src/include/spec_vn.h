/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#ifndef SRC_INCLUDE_SPEC_VN_H_
#define SRC_INCLUDE_SPEC_VN_H_

#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>

namespace Smash {
    class Spec_vn {
        public:
            // dN/dY, dN/dEta for charged particles
            // dN/(2pi dYPtdPt) and dN/(2pi dEta PtdPt)
            // v2, v3 and v4 for charged particles
            // and pion+, kaon+ and proton

        Spec_vn(std::string fpath_out, int vn_method);

        void dndy_(std::string hadron_type);

        void dndpt_(std::string hadron_type);

        void vn_(std::string hadron_type);
    };
}


#endif
