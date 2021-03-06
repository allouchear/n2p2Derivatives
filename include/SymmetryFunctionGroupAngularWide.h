// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef SYMMETRYFUNCTIONGROUPANGULARWIDE_H
#define SYMMETRYFUNCTIONGROUPANGULARWIDE_H

#include "SymmetryFunctionGroup.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

#include "Derivatives.h"

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunction;
class SymmetryFunctionAngularWide;

/** Angular symmetry function group (type 3)
 *
 * @f[
   G^9_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
           \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
           \mathrm{e}^{-\eta( (r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 ) }
           f_c(r_{ij}) f_c(r_{ik}) 
 * @f]
 * Common features:
 * - element of central atom
 * - element of neighbor atom 1
 * - element of neighbor atom 2
 * - cutoff type
 * - @f$r_c@f$
 * - @f$\alpha@f$
 */
class SymmetryFunctionGroupAngularWide : public SymmetryFunctionGroup
{
public:
    /** Constructor, sets type = 3
     */
    SymmetryFunctionGroupAngularWide(ElementMap const& elementMap);
    /** Overload == operator.
     */
    bool operator==(SymmetryFunctionGroup const& rhs) const;
    /** Overload != operator.
     */
    bool operator!=(SymmetryFunctionGroup const& rhs) const;
    /** Overload < operator.
     */
    bool operator<(SymmetryFunctionGroup const& rhs) const;
    /** Overload > operator.
     */
    bool operator>(SymmetryFunctionGroup const& rhs) const;
    /** Overload <= operator.
     */
    bool operator<=(SymmetryFunctionGroup const& rhs) const;
    /** Overload >= operator.
     */
    bool operator>=(SymmetryFunctionGroup const& rhs) const;
    /** Potentially add a member to group.
     *
     * @param[in] symmetryFunction Candidate symmetry function.
     * @return If addition was successful.
     *
     * If symmetry function is compatible with common feature list its pointer
     * will be added to #members.
     */
    bool addMember(SymmetryFunction const* const symmetryFunction);
    /** Sort member symmetry functions.
     *
     * Also allocate and precalculate additional stuff.
     */
    void sortMembers();
    /** Fill #scalingFactors with values from member symmetry functions.
     */
    void setScalingFactors();
    /** Calculate all symmetry functions of this group for one atom.
     *
     * @param[in,out] atom Atom for which symmetry functions are caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    void calculate(Atom& atom, bool const derivatives) const;
#ifdef HIGH_DERIVATIVES
    /** compute second derivatives one Function on one atom.
     *
     * @param[in,out] atom Atom for which the derivatives of symmetry function are caluclated.
     * @param[in] deriv  contain the derivative of G by xx, xy, xz, yy, yz, zz coordinates of atom
     */
    void         compute2Derivatives(Atom& atom, std::vector< std::vector<double> >& deriv) const;
    /** Calculate derivatives of symmetry function on one atom.
     * The maxorder derivatives is defined in deriv as input
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[out] derivatives saved in deriv
     */
    void   	computeDerivatives(Atom& atom, Derivatives& deriv) const;
#endif
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    std::vector<std::string>
         parameterLines() const;

private:
    /// Element index of neighbor atom 1 (common feature).
    std::size_t                                     e1;
    /// Element index of neighbor atom 2 (common feature).
    std::size_t                                     e2;
    /// Vector of all group member pointers.
    std::vector<SymmetryFunctionAngularWide const*> members;
    /// Vector indicating whether exponential term needs to be calculated.
    std::vector<bool>                               calculateExp;
    /// Vector containing precalculated normalizing factor for each zeta.
    std::vector<double>                             factorNorm;
    /// Vector containing precalculated normalizing factor for derivatives.
    std::vector<double>                             factorDeriv;
    /// Vector containing values of all member symmetry functions.
    std::vector<bool>                               useIntegerPow;
    /// Vector containing values of all member symmetry functions.
    std::vector<size_t>                             memberIndex;
    /// Vector containing values of all member symmetry functions.
    std::vector<int>                                zetaInt;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                             eta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                             zeta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                             lambda;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                             zetaLambda;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                               rs;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionGroupAngularWide::
operator!=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionGroupAngularWide::
operator>(SymmetryFunctionGroup const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionGroupAngularWide::
operator<=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionGroupAngularWide::
operator>=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) < rhs);
}

}

#endif
