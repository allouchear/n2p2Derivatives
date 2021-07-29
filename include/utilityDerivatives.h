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

#ifndef UTILITYDERIVATIVES_H
#define UTILITYDERIVATIVES_H

#ifdef HIGH_DERIVATIVES

#include <cstdio>    // FILE
#include <fstream>   // std::ofstream
#include <map>       // std::map
#include <sstream>   // std::stringstream
#include <stdexcept> // std::range_error
#include <string>    // std::string
#include <vector>    // std::vector
#include "CutoffFunction.h"

namespace nnp
{
void permutation(int *arr, int start, int end, double val, double*** tab3);
void permutation(int *arr, int start, int end, double val, double**** tab4);
/** Compute d^nR/dxi^n
 *
 * @param[in] dr : ri-rj
 * @param[in] r 
 * @param[in] num coordinante
 * @param[in] order n
 *
 * @return derivative
 */
double getDerivativesR(const double* dr, const double& r, int ialpha, int order);
/** Compute d^nR/dxi^ni^nj
 *
 * @param[in] dr : ri-rj
 * @param[in] r 
 * @param[in] num  of first coordinante
 * @param[in] order  of the first coordinate
 * @param[in] num  of second coordinante
 * @param[in] order  of the second coordinate
 *
 * @return derivative
 */
double getDerivativesR(const double* dr, const double& r, int ialpha, int orderi, int jalpha, int orderj);

double getDerivativesRadial(double x, double eta, int order);
double getDerivativesG(const double* rijv, double eta, double rs, double secondDeriv[][6], const CutoffFunction& cf);

/** compute d/dxl Rijk = Rij*Rik 
 * @param[in] rij 
 * @param[in] rik 
 * @param[out] firstDeriv vector with 9 elements
 */
double getDerivativesRijk(const double* rij, const double* rik, double firstDeriv[]);
/* secondDeriv matrix with 9x9 elements */
/** compute d2/dxldxm of Rijk  = Rij*Rik
 * @param[out] matrix with 9x9 elements
 */
void getDerivativesRijk(double secondDeriv[][9]);
/** compute d/dxl &&  d2/dxldxm of Rijk  = Rij*Rik
 * @param[in] rij 
 * @param[in] rik 
 * @param[out] firstDeriv vector with 9 elements
 * @param[out] matrix with 9x9 elements
 * @return vec rij dot vec rik
 */
double getDerivativesRijk(const double* rij, const double* rik, double firstDeriv[],double secondDeriv[][9]);
/* firstDeriv vector with 3 elements */
/* secondDeriv matrix with 3x3 elements */
double getDerivativesRij(const double* rij,  double firstDeriv[]);
double getDerivativesRij(const double* rij,  double secondDeriv[][6]);
double getDerivativesRij(const double* rij,  double firstDeriv[], double secondDeriv[][6]);
double getDerivativesRijRik(const double* rij, const double* rik, double firstDeriv[]);
double getDerivativesRijRik(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9]);
double getDerivativesRijRikm1(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9]);
// derivatives of cos theatijk
double getDerivativescosijk(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9]);
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
/* derivatives of Rij^2 + Rik^2 */
double getDerivativesRij2pRik2(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9]);
double getDerivativesexpetaRij2pRik2(const double* rij, const double* rik,double eta, double rs, double firstDeriv[],double secondDeriv[][9]);
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
/* derivatives of Rij^2 + Rik^2 + Rjk^2 */
double getDerivativesRij2pRik2pRjk2(const double* rij, const double* rik, const double* rjk, double firstDeriv[],double secondDeriv[][9]);
double getDerivativesexpetaRij2pRik2pRjk2(const double* rij, const double* rik, const double* rjk, double eta, double rs, double firstDeriv[],double secondDeriv[][9]);

//typedef double (*pgetfDerivatives)(const double* rij, double df[], double d2f[][6]);
double getDerivativesfc(const double* rij, const double* rik, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf);
double getDerivativesfc(const double* rij, const double* rik, const double* rjk, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf);
double getDerivativesG(const double* rij, const double* rik, double lambda, double zeta, double eta, double rs, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf);
double getDerivativesG(const double* rij, const double* rik, const double* rjk, double lambda, double zeta, double eta, double rs, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf);
void setSecondDerivesijk(double secondDeriv[][9], int indexI, int indexJ, int indexK, double scalingFactor, std::vector< std::vector<double> >& deriv);
void setSecondDerivesij(double secondDeriv[][6], int indexI, int indexJ, double scalingFactor, std::vector< std::vector<double> >& deriv);
void toPhysicalUnits(double* deriv, size_t dervSize, double convEnergy, double convLength);


double* new1Dtable(int size);
double* copy1Dtable(int size, double* t);
void free1Dtable(double* t);
double** new2Dtable(int size);
double** copy2Dtable(int size, double** t);
void free2Dtable(int size, double**t);
double*** new3Dtable(int size);
double*** copy3Dtable(int size, double*** t);
void free3Dtable(int size, double***t);
double**** new4Dtable(int size);
double**** copy4Dtable(int size, double**** t);
void free4Dtable(int size, double****t);
}
#endif /* HIGH_DERIVATIVES */

#endif
