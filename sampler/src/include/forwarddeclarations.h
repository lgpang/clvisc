/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#ifndef SRC_INCLUDE_FORWARDDECLARATIONS_H_
#define SRC_INCLUDE_FORWARDDECLARATIONS_H_

#include <iosfwd>

// the forward declarations should not appear in doxygen output
#ifndef DOXYGEN

#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#endif

template <typename T>
class allocator;
template <typename T, typename A>
class vector;

template <typename T>
struct default_delete;
template <typename T, typename Deleter>
class unique_ptr;

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
}  // namespace std
#endif

namespace Smash {

template <typename T>
using build_unique_ptr_ = std::unique_ptr<T, std::default_delete<T>>;
template <typename T>
using build_vector_ = std::vector<T, std::allocator<T>>;

/** Initial condition for a particle in a box.
*
* If PeakedMomenta is used, all particles have the same momentum
* \f$p = 3 \cdot T\f$ with T the temperature.
*
* Else, a thermalized ensemble is generated (the momenta are sampled
* from a Maxwell-Boltzmann distribution).
*
* In either case, the positions in space are chosen randomly.
*/

}  // namespace Smash

#endif  // DOXYGEN
#endif  // SRC_INCLUDE_FORWARDDECLARATIONS_H_
