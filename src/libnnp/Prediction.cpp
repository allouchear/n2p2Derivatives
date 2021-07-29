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

#include "Prediction.h"
#include <fstream>   // std::ifstream
#include <stdexcept> // std::runtime_error
#include "Stopwatch.h"
#include "utility.h"
#include "utilityDerivatives.h"
#include "Derivatives.h"
#include <iostream>   // std::iostream

using namespace std;
using namespace nnp;

Prediction::Prediction() : Mode(),
                           fileNameSettings  ("input.nn"          ),
                           fileNameScaling   ("scaling.data"      ),
                           formatWeightsFiles("weights.%03zu.data")
{
}

void Prediction::setup()
{
    initialize();
    loadSettingsFile(fileNameSettings);
    setupGeneric();
    setupSymmetryFunctionScaling(fileNameScaling);
    setupNeuralNetworkWeights(formatWeightsFiles);
    setupSymmetryFunctionStatistics(false, false, true, false);
}
double Prediction::getConvLength()
{
    return convLength;
}

void Prediction::readStructureFromFile(string const& fileName)
{
    ifstream file;
    file.open(fileName.c_str());
    structure.reset();
    structure.setElementMap(elementMap);
    structure.readFromFile(fileName);
    removeEnergyOffset(structure);
    file.close();

    return;
}

void Prediction::predict()
{
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    structure.calculateNeighborList(maxCutoffRadius);
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicNeuralNetworks(structure, true);
    calculateEnergy(structure);
    calculateForces(structure);
    if (normalize)
    {
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    addEnergyOffset(structure, false);
    addEnergyOffset(structure, true);

    return;
}
#ifdef HIGH_DERIVATIVES
Derivatives Prediction::getHighDerivatives(int order, int method)
{
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    if(!structure.hasNeighborList)
    	structure.calculateNeighborList(maxCutoffRadius);
    if(!structure.hasSymmetryFunctions)
    {
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    }
    calculateAtomicNeuralNetworks(structure, true);
    calculateEnergy(structure);
    calculateForces(structure);
    Derivatives deriv;
    deriv = computeHighDerivatives(structure, order, method);
    if (normalize)
    {
	deriv.toPhysicalUnits(convEnergy, convLength);
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    addEnergyOffset(structure, false);
    addEnergyOffset(structure, true);
    return deriv;
}
double* Prediction::getHessian()
{
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    if(!structure.hasNeighborList)
    	structure.calculateNeighborList(maxCutoffRadius);
    if(!structure.hasSymmetryFunctions)
    {
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    }
    calculateAtomicNeuralNetworks(structure, true);
    calculateEnergy(structure);
    calculateForces(structure);
    size_t derivSize;
    double* deriv = computeHessian(structure, derivSize);
    if (normalize)
    {
	toPhysicalUnits(deriv, derivSize, convEnergy, convLength);
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);
    }
    addEnergyOffset(structure, false);
    addEnergyOffset(structure, true);
    return deriv;
}
#endif
