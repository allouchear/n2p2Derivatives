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

#include "SymmetryFunctionWeightedAngular.h"
#include "Atom.h"
#include "ElementMap.h"
#include "utility.h"
#include "utilityDerivatives.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp, pow, cos
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <iostream> 

using namespace std;
using namespace nnp;

SymmetryFunctionWeightedAngular::
SymmetryFunctionWeightedAngular(ElementMap const& elementMap) :
    SymmetryFunction(13, elementMap),
    useIntegerPow(false),
    zetaInt      (0    ),
    eta          (0.0  ),
    rs           (0.0  ),
    lambda       (0.0  ),
    zeta         (0.0  )
{
    minNeighbors = 2;
    parameters.insert("rs");
    parameters.insert("eta");
    parameters.insert("zeta");
    parameters.insert("lambda");
}

bool SymmetryFunctionWeightedAngular::
operator==(SymmetryFunction const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionWeightedAngular const& c =
        dynamic_cast<SymmetryFunctionWeightedAngular const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (rs          != c.rs         ) return false;
    if (zeta        != c.zeta       ) return false;
    if (lambda      != c.lambda     ) return false;
    return true;
}

bool SymmetryFunctionWeightedAngular::
operator<(SymmetryFunction const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionWeightedAngular const& c =
        dynamic_cast<SymmetryFunctionWeightedAngular const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (eta         < c.eta        ) return true;
    else if (eta         > c.eta        ) return false;
    if      (rs          < c.rs         ) return true;
    else if (rs          > c.rs         ) return false;
    if      (zeta        < c.zeta       ) return true;
    else if (zeta        > c.zeta       ) return false;
    if      (lambda      < c.lambda     ) return true;
    else if (lambda      > c.lambda     ) return false;
    return false;
}

void SymmetryFunctionWeightedAngular::
     setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec     = elementMap[splitLine.at(0)];
    eta    = atof(splitLine.at(2).c_str());
    rs     = atof(splitLine.at(3).c_str());
    lambda = atof(splitLine.at(4).c_str());
    zeta   = atof(splitLine.at(5).c_str());
    rc     = atof(splitLine.at(6).c_str());

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    zetaInt = round(zeta);
    if (fabs(zeta - zetaInt) <= numeric_limits<double>::min())
    {
        useIntegerPow = true;
    }
    else
    {
        useIntegerPow = false;
    }

    return;
}

void SymmetryFunctionWeightedAngular::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    eta /= convLength * convLength;
    rs *= convLength;
    rc *= convLength;

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

string SymmetryFunctionWeightedAngular::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %16.8E %16.8E %16.8E "
                     "%16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     eta * convLength * convLength,
                     rs / convLength,
                     lambda,
                     zeta,
                     rc / convLength);

    return s;
}

void SymmetryFunctionWeightedAngular::calculate(Atom&      atom,
                                                bool const derivatives) const
{
    double const pnorm  = pow(2.0, 1.0 - zeta);
    double const pzl    = zeta * lambda;
    double const rc2    = rc * rc;
    double       result = 0.0;

    size_t numNeighbors = atom.numNeighbors;
    // Prevent problematic condition in loop test below (j < numNeighbors - 1).
    if (numNeighbors == 0) numNeighbors = 1;

    for (size_t j = 0; j < numNeighbors - 1; j++)
    {
        Atom::Neighbor& nj = atom.neighbors[j];
        size_t const nej = nj.element;
        double const rij = nj.d;
        if (rij < rc)
        {
            double const r2ij   = rij * rij;

            // Calculate cutoff function and derivative.
#ifdef NOCFCACHE
            double pfcij;
            double pdfcij;
            fc.fdf(rij, pfcij, pdfcij);
#else
            // If cutoff radius matches with the one in the neighbor storage
            // we can use the previously calculated value.
            double& pfcij = nj.fc;
            double& pdfcij = nj.dfc;
            if (nj.cutoffType != cutoffType ||
                nj.rc != rc ||
                nj.cutoffAlpha != cutoffAlpha)
            {
                fc.fdf(rij, pfcij, pdfcij);
                nj.rc = rc;
                nj.cutoffType = cutoffType;
                nj.cutoffAlpha = cutoffAlpha;
            }
#endif
            for (size_t k = j + 1; k < numNeighbors; k++)
            {
                Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = nk.element;
                double const rik = nk.d;
                if (rik < rc)
                {
                    Vec3D drjk = nk.dr - nj.dr;
                    double rjk = drjk.norm2();;
                    if (rjk < rc2)
                    {
                        // Energy calculation.
#ifdef NOCFCACHE
                        double pfcik;
                        double pdfcik;
                        fc.fdf(rik, pfcik, pdfcik);
#else
                        double& pfcik = nk.fc;
                        double& pdfcik = nk.dfc;
                        if (nk.cutoffType != cutoffType ||
                            nk.rc != rc ||
                            nk.cutoffAlpha != cutoffAlpha)
                        {
                            fc.fdf(rik, pfcik, pdfcik);
                            nk.rc = rc;
                            nk.cutoffType = cutoffType;
                            nk.cutoffAlpha = cutoffAlpha;
                        }
#endif
                        rjk = sqrt(rjk);
                        double pfcjk;
                        double pdfcjk;
                        fc.fdf(rjk, pfcjk, pdfcjk);

                        Vec3D drij = nj.dr;
                        Vec3D drik = nk.dr;
                        double costijk = drij * drik;;
                        double rinvijik = 1.0 / rij / rik;
                        costijk *= rinvijik;

                        double const pfc = pfcij * pfcik * pfcjk;
                        double const r2ik = rik * rik;
                        double const rijs = rij - rs;
                        double const riks = rik - rs;
                        double const rjks = rjk - rs;
                        double const pexp = elementMap.atomicNumber(nej)
                                          * elementMap.atomicNumber(nek)
                                          * exp(-eta * (rijs * rijs +
                                                        riks * riks +
                                                        rjks * rjks));
                        double const plambda = 1.0 + lambda * costijk;
                        double       fg = pexp;
                        if (plambda <= 0.0) fg = 0.0;
                        else
                        {
                            if (useIntegerPow)
                            {
                                fg *= pow_int(plambda, zetaInt - 1);
                            }
                            else
                            {
                                fg *= pow(plambda, zeta - 1.0);
                            }
                        }
                        result += fg * plambda * pfc;

                        // Force calculation.
                        if (!derivatives) continue;
                        fg       *= pnorm;
                        rinvijik *= pzl;
                        costijk  *= pzl;
                        double const p2etapl = 2.0 * eta * plambda;
                        double const p1 = fg * (pfc * (rinvijik - costijk
                                        / r2ij - p2etapl * rijs / rij) + pfcik
                                        * pfcjk * pdfcij * plambda / rij);
                        double const p2 = fg * (pfc * (rinvijik - costijk
                                        / r2ik - p2etapl * riks / rik) + pfcij
                                        * pfcjk * pdfcik * plambda / rik);
                        double const p3 = fg * (pfc * (rinvijik + p2etapl
                                        * rjks / rjk) - pfcij * pfcik * pdfcjk
                                        * plambda / rjk);
                        drij *= p1 * scalingFactor;
                        drik *= p2 * scalingFactor;
                        drjk *= p3 * scalingFactor;

                        // Save force contributions in Atom storage.
                        atom.dGdr[index] += drij + drik;
                        nj.dGdr[index]   -= drij + drjk;
                        nk.dGdr[index]   -= drik - drjk;
                    } // rjk <= rc
                } // rik <= rc
            } // k
        } // rij <= rc
    } // j
    result *= pnorm;

    atom.G[index] = scale(result);

    return;
}
#ifdef HIGH_DERIVATIVES
// deriv is a 3Nx3N matrix where N = # of atomes
// deriv matrix must be initialized before calling of compute2Derivatives method.
// order is already defined in deriv.maxOrder
void  SymmetryFunctionWeightedAngular::computeDerivatives(Atom& atom, Derivatives& deriv) const
{
    	double const rc2    = rc * rc;
    	Derivatives dum(deriv.getMaxOrder(), 9);
	deriv.reset();
	size_t numNeighbors = atom.numNeighbors;
        // Prevent problematic condition in loop test below (j < numNeighbors - 1).
        if (numNeighbors == 0) numNeighbors = 1;
	for (size_t j = 0; j < numNeighbors - 1; j++)
	{
        	Atom::Neighbor& nj = atom.neighbors[j];
        	size_t const nej = nj.element;
        	double const rij = nj.d;
        	if (rij < rc)
        	{
            		for (size_t k = j + 1; k < numNeighbors; k++)
            		{
                		Atom::Neighbor& nk = atom.neighbors[k];
                		size_t const nek = nk.element;
        			double const rik = nk.d;
                    		if (rik < rc)
                    		{
					double vrjk[3];
					for(size_t ia=0;ia<3;ia++) vrjk[ia] = nk.dr.r[ia] - nj.dr.r[ia];
					double rjk = 0;
					for(size_t ia=0;ia<3;ia++) rjk += vrjk[ia]*vrjk[ia];
					if (rjk < rc2)
					{
						dum.computeG(zeta, lambda, eta, nj.dr.r, nk.dr.r,  vrjk,  rs, fc);
						double scalf = elementMap.atomicNumber(nej)*elementMap.atomicNumber(nek)*scalingFactor;
						deriv.addDerives(dum, atom.index,  nj.index, nk.index , scalf);
					}
                    		} // rik <= rc
            		} // k
        	} // rij <= rc
	} // j
}
// deriv is a 3Nx3N matrix where N = # of atomes
// deriv matrix must be initialized before calling of compute2Derivatives method.
void  SymmetryFunctionWeightedAngular::compute2Derivatives(Atom& atom, std::vector< std::vector<double> >& deriv) const
{
    	double const rc2    = rc * rc;
	for (size_t i = 0; i < deriv.size(); ++i)
	for (size_t j = 0; j < deriv[i].size(); ++j) deriv[i][j] = 0.0;
	double firstDeriv[9];
	double secondDeriv[9][9];
	size_t numNeighbors = atom.numNeighbors;
        // Prevent problematic condition in loop test below (j < numNeighbors - 1).
        if (numNeighbors == 0) numNeighbors = 1;
	for (size_t j = 0; j < numNeighbors - 1; j++)
	{
        	Atom::Neighbor& nj = atom.neighbors[j];
        	size_t const nej = nj.element;
        	double const rij = nj.d;
        	if (rij < rc)
        	{
            		for (size_t k = j + 1; k < numNeighbors; k++)
            		{
                		Atom::Neighbor& nk = atom.neighbors[k];
                		size_t const nek = nk.element;
        			double const rik = nk.d;
                    		if (rik < rc)
                    		{
						double vrjk[3];
						for(size_t ia=0;ia<3;ia++) vrjk[ia] = nk.dr.r[ia] - nj.dr.r[ia];
						double rjk = 0;
						for(size_t ia=0;ia<3;ia++) rjk += vrjk[ia]*vrjk[ia];
						if (rjk < rc2)
						{
						getDerivativesG(nj.dr.r, nk.dr.r, vrjk, lambda, zeta, eta, rs, firstDeriv, secondDeriv, fc);
						double scalf = elementMap.atomicNumber(nej)*elementMap.atomicNumber(nek)*scalingFactor;
						setSecondDerivesijk(secondDeriv, atom.index, nj.index, nk.index, scalf, deriv);
						}
                    		} // rik <= rc
            		} // k
        	} // rij <= rc
	} // j
}
#endif

string SymmetryFunctionWeightedAngular::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 eta * convLength * convLength,
                 rs / convLength,
                 lambda,
                 zeta,
                 rc / convLength,
                 (int)cutoffType,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymmetryFunctionWeightedAngular::parameterInfo() const
{
    vector<string> v = SymmetryFunction::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(),
                      eta * convLength * convLength));
    s = "lambda";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), lambda));
    s = "zeta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), zeta));
    s = "rs";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rs / convLength));

    return v;
}

double SymmetryFunctionWeightedAngular::calculateRadialPart(
                                                         double distance) const
{
    double const& r = distance * convLength;
    double const p = exp(-eta * (r - rs) * (r - rs)) * fc.f(r);

    return p * p * p;
}

double SymmetryFunctionWeightedAngular::calculateAngularPart(double angle) const
{
    return 2.0 * pow((1.0 + lambda * cos(angle)) / 2.0, zeta);
}
