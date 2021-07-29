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
// Created by AR. Allouche 04/01/2020

#include "PredictionES.h"
#include <fstream>   // std::ifstream
#include <stdexcept> // std::runtime_error
#include "Stopwatch.h"
#include "utility.h"
#include "utilityDerivatives.h"
#include "Derivatives.h"
#include <iostream>   // std::iostream


using namespace std;
using namespace nnp;

PredictionES::PredictionES() : Mode(),
                           fileNameSettings  ("inputES.nn"          ),
                           formatWeightsFiles("weightsES.%03zu.data")
{
}

void PredictionES::setup()
{
    initialize();
    loadSettingsFile(fileNameSettings);
    setupGeneric();
    setupSymmetryFunctionScalingNone();
    setupNeuralNetworkWeights(formatWeightsFiles);
    setupSymmetryFunctionStatistics(false, false, false, false);
}

void PredictionES::readStructureFromFile(string const& fileName)
{
    ifstream file;
    file.open(fileName.c_str());
    structure.reset();
    structure.setElementMap(elementMap);
    structure.readFromFile(fileName);
    file.close();

    return;
}

void PredictionES::predict()
{
    structure.calculateNeighborList(maxCutoffRadius);
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, false);
#else
    calculateSymmetryFunctionGroups(structure, false);
#endif
    calculateAtomicChargesNeuralNetworks(structure, false);
    calculateCharge(structure);
    calculateDipole(structure);

    return;
}
double** PredictionES::getdDipole()
{
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    if(!structure.hasNeighborList) structure.calculateNeighborList(maxCutoffRadius);
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicChargesNeuralNetworks(structure, true);
    calculateCharge(structure);
    calculateDipole(structure);

    double** deriv = NULL;
    deriv = computedDipole(structure);
    if (normalize && deriv)
    {
    	double cv = convLength/convDipole;
	size_t nAtoms3 = 3*structure.numAtoms;

    	for(int c=0;c<3;c++)
    	for(size_t i=0;i<nAtoms3;i++)
		deriv[c][i] *= cv;
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    return deriv;
}
#ifdef HIGH_DERIVATIVES
double** PredictionES::getd2Dipole()
{
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    if(!structure.hasNeighborList) structure.calculateNeighborList(maxCutoffRadius);
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicChargesNeuralNetworks(structure, true);
    calculateCharge(structure);
    calculateDipole(structure);

    size_t dsize = 0;
    double** deriv = NULL;
    deriv = computed2Dipole(structure, dsize);
    if (normalize && deriv)
    {
    	double cv = convLength*convLength/convDipole;

    	for(int c=0;c<3;c++)
    	for(size_t i=0;i<dsize;i++)
		deriv[c][i] *= cv;
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    return deriv;
}
Derivatives* PredictionES::getHighDerivatives(int order, int method)
{
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    if(!structure.hasNeighborList) structure.calculateNeighborList(maxCutoffRadius);
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicChargesNeuralNetworks(structure, true);
    calculateCharge(structure);
    calculateDipole(structure);

    Derivatives* deriv;
    deriv = computeHighDerivativesDipole(structure, order, method);

    if (normalize && deriv)
    {
    	for(int c=0;c<3;c++)
		deriv[c].toPhysicalUnits(convDipole, convLength);

        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    return deriv;
}
#endif
