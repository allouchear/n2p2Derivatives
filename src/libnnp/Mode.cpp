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

#include "Mode.h"
#include "NeuralNetwork.h"
#include "utility.h"
#include "utilityDerivatives.h"
#include "version.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm> // std::min, std::max
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <iostream> 
#include <sstream> 

using namespace std;
using namespace nnp;

Mode::Mode() : normalize                 (false),
               checkExtrapolationWarnings(false),
               numElements               (0    ),
               maxCutoffRadius           (0.0  ),
               cutoffAlpha               (0.0  ),
               meanEnergy                (0.0  ),
               convEnergy                (1.0  ),
               convLength                (1.0  ),
               convDipole                (1.0  )
{
               dftd3 = DFTD3();
}

void Mode::initialize()
{

    log << "\n";
    log << "*****************************************"
           "**************************************\n";
    log << "\n";
    log << "   NNP LIBRARY v" NNP_VERSION "\n";
    log << "   ------------------\n";
    log << "\n";
    log << "Git branch  : " NNP_GIT_BRANCH "\n";
    log << "Git revision: " NNP_GIT_REV_SHORT " (" NNP_GIT_REV ")\n";
    log << "\n";
#ifdef _OPENMP
    log << strpr("Number of OpenMP threads: %d", omp_get_max_threads());
    log << "\n";
#endif
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::loadSettingsFile(string const& fileName)
{
    log << "\n";
    log << "*** SETUP: SETTINGS FILE ****************"
           "**************************************\n";
    log << "\n";

    settings.loadFile(fileName);
    log << settings.info();

    log << "*****************************************"
           "**************************************\n";

    return;
}
void Mode::setupDFTD3()
{
    if (settings.keywordExists("dftd3_func"))
    {
	dftd3 = DFTD3(settings["dftd3_func"]);
    }
    else dftd3 = DFTD3();
}

void Mode::setupGeneric()
{
    setupNormalization();
    setupElementMap();
    setupDFTD3();
    setupElements();
    setupCutoff();
    setupSymmetryFunctions();
#ifndef NOSFGROUPS
    setupSymmetryFunctionGroups();
#endif
    setupNeuralNetwork();

    return;
}

void Mode::setupNormalization()
{
    log << "\n";
    log << "*** SETUP: NORMALIZATION ****************"
           "**************************************\n";
    log << "\n";

    if (settings.keywordExists("mean_energy") &&
        settings.keywordExists("conv_energy") &&
        settings.keywordExists("conv_length"))
    {
        normalize = true;
        meanEnergy = atof(settings["mean_energy"].c_str());
        convEnergy = atof(settings["conv_energy"].c_str());
        convLength = atof(settings["conv_length"].c_str());
        if(settings.keywordExists("conv_dipole"))
        	convDipole = atof(settings["conv_dipole"].c_str());
        log << "Data set normalization is used.\n";
        log << strpr("Mean energy per atom     : %24.16E\n", meanEnergy);
        log << strpr("Conversion factor energy : %24.16E\n", convEnergy);
        log << strpr("Conversion factor length : %24.16E\n", convLength);
        log << strpr("Conversion factor dipole : %24.16E\n", convDipole);
        if (settings.keywordExists("atom_energy"))
        {
            log << "\n";
            log << "Atomic energy offsets are used in addition to"
                   " data set normalization.\n";
            log << "Offsets will be subtracted from reference energies BEFORE"
                   " normalization is applied.\n";
        }
    }
    else if ((!settings.keywordExists("mean_energy")) &&
             (!settings.keywordExists("conv_energy")) &&
             (!settings.keywordExists("conv_length")))
    {
        normalize = false;
        log << "Data set normalization is not used.\n";
    }
    else
    {
        throw runtime_error("ERROR: Incorrect usage of normalization"
                            " keywords.\n"
                            "       Use all or none of \"mean_energy\", "
                            "\"conv_energy\" and \"conv_length\".\n");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupElementMap()
{
    log << "\n";
    log << "*** SETUP: ELEMENT MAP ******************"
           "**************************************\n";
    log << "\n";

    elementMap.registerElements(settings["elements"]);
    log << strpr("Number of element strings found: %d\n", elementMap.size());
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        log << strpr("Element %2zu: %2s (%3zu)\n", i, elementMap[i].c_str(),
                     elementMap.atomicNumber(i));
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupElements()
{
    log << "\n";
    log << "*** SETUP: ELEMENTS *********************"
           "**************************************\n";
    log << "\n";

    numElements = (size_t)atoi(settings["number_of_elements"].c_str());
    if (numElements != elementMap.size())
    {
        throw runtime_error("ERROR: Inconsistent number of elements.\n");
    }
    log << strpr("Number of elements is consistent: %zu\n", numElements);

    for (size_t i = 0; i < numElements; ++i)
    {
        elements.push_back(Element(i, elementMap));
    }

    if (settings.keywordExists("atom_energy"))
    {
        Settings::KeyRange r = settings.getValues("atom_energy");
        for (Settings::KeyMap::const_iterator it = r.first;
             it != r.second; ++it)
        {
            vector<string> args    = split(reduce(it->second.first));
            size_t         element = elementMap[args.at(0)];
            elements.at(element).
                setAtomicEnergyOffset(atof(args.at(1).c_str()));
        }
    }
    log << "Atomic energy offsets per element:\n";
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        log << strpr("Element %2zu: %16.8E\n",
                     i, elements.at(i).getAtomicEnergyOffset());
    }

    log << "Energy offsets are automatically subtracted from reference "
           "energies.\n";
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupCutoff()
{
    log << "\n";
    log << "*** SETUP: CUTOFF FUNCTIONS *************"
           "**************************************\n";
    log << "\n";

    vector<string> args = split(settings["cutoff_type"]);

    cutoffType = (CutoffFunction::CutoffType) atoi(args.at(0).c_str());
    if (args.size() > 1)
    {
        cutoffAlpha = atof(args.at(1).c_str());
        if (0.0 < cutoffAlpha && cutoffAlpha >= 1.0)
        {
            throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
        }
    }
    log << strpr("Parameter alpha for inner cutoff: %f\n", cutoffAlpha);
    log << "Inner cutoff = Symmetry function cutoff * alpha\n";

    log << "Equal cutoff function type for all symmetry functions:\n";
    if (cutoffType == CutoffFunction::CT_COS)
    {
        log << strpr("CutoffFunction::CT_COS (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = 1/2 * (cos(pi*x) + 1)\n";
    }
    else if (cutoffType == CutoffFunction::CT_TANHU)
    {
        log << strpr("CutoffFunction::CT_TANHU (%d)\n", cutoffType);
        log << "f(r) = tanh^3(1 - r/rc)\n";
        if (cutoffAlpha > 0.0)
        {
            log << "WARNING: Inner cutoff parameter not used in combination"
                   " with this cutoff function.\n";
        }
    }
    else if (cutoffType == CutoffFunction::CT_TANH)
    {
        log << strpr("CutoffFunction::CT_TANH (%d)\n", cutoffType);
        log << "f(r) = c * tanh^3(1 - r/rc), f(0) = 1\n";
        if (cutoffAlpha > 0.0)
        {
            log << "WARNING: Inner cutoff parameter not used in combination"
                   " with this cutoff function.\n";
        }
    }
    else if (cutoffType == CutoffFunction::CT_POLY1)
    {
        log << strpr("CutoffFunction::CT_POLY1 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (2x - 3)x^2 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY2)
    {
        log << strpr("CutoffFunction::CT_POLY2 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = ((15 - 6x)x - 10)x^3 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY3)
    {
        log << strpr("CutoffFunction::CT_POLY3 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY4)
    {
        log << strpr("CutoffFunction::CT_POLY4 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_EXP)
    {
        log << strpr("CutoffFunction::CT_EXP (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = exp(-1 / 1 - x^2)\n";
    }
    else if (cutoffType == CutoffFunction::CT_HARD)
    {
        log << strpr("CutoffFunction::CT_HARD (%d)\n", cutoffType);
        log << "f(r) = 1\n";
        log << "WARNING: Hard cutoff used!\n";
    }
    else
    {
        throw invalid_argument("ERROR: Unknown cutoff type.\n");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctions()
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTIONS ***********"
           "**************************************\n";
    log << "\n";

    Settings::KeyRange r = settings.getValues("symfunction_short");
    for (Settings::KeyMap::const_iterator it = r.first; it != r.second; ++it)
    {
        vector<string> args    = split(reduce(it->second.first));
        size_t         element = elementMap[args.at(0)];

        elements.at(element).addSymmetryFunction(it->second.first,
                                                 it->second.second);
    }

    log << "Abbreviations:\n";
    log << "--------------\n";
    log << "ind .... Symmetry function index.\n";
    log << "ec ..... Central atom element.\n";
    log << "ty ..... Symmetry function type.\n";
    log << "e1 ..... Neighbor 1 element.\n";
    log << "e2 ..... Neighbor 2 element.\n";
    log << "eta .... Gaussian width eta.\n";
    log << "rs ..... Shift distance of Gaussian.\n";
    log << "la ..... Angle prefactor lambda.\n";
    log << "zeta ... Angle term exponent zeta.\n";
    log << "rc ..... Cutoff radius.\n";
    log << "ct ..... Cutoff type.\n";
    log << "ca ..... Cutoff alpha.\n";
    log << "ln ..... Line number in settings file.\n";
    log << "\n";
    maxCutoffRadius = 0.0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        if (normalize) it->changeLengthUnitSymmetryFunctions(convLength);
        it->sortSymmetryFunctions();
        maxCutoffRadius = max(it->getMaxCutoffRadius(), maxCutoffRadius);
        it->setCutoffFunction(cutoffType, cutoffAlpha);
        log << strpr("Short range atomic symmetry functions element %2s :\n",
                     it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind ec ty e1 e2       eta        rs la "
               "zeta        rc ct   ca    ln\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionParameters();
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }
    minNeighbors.resize(numElements, 0);
    minCutoffRadius.resize(numElements, maxCutoffRadius);
    for (size_t i = 0; i < numElements; ++i)
    {
        minNeighbors.at(i) = elements.at(i).getMinNeighbors();
        minCutoffRadius.at(i) = elements.at(i).getMinCutoffRadius();
        log << strpr("Minimum cutoff radius for element %2s: %f\n",
                     elements.at(i).getSymbol().c_str(),
                     minCutoffRadius.at(i) / convLength);
    }
    log << strpr("Maximum cutoff radius (global)      : %f\n",
                 maxCutoffRadius / convLength);

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionScalingNone()
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION SCALING ****"
           "**************************************\n";
    log << "\n";

    log << "No scaling for symmetry functions.\n";
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->setScalingNone();
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionScaling(string const& fileName)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION SCALING ****"
           "**************************************\n";
    log << "\n";

    log << "Equal scaling type for all symmetry functions:\n";
    if (   ( settings.keywordExists("scale_symmetry_functions" ))
        && (!settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymmetryFunction::ST_SCALE;
        log << strpr("Scaling type::ST_SCALE (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmin) / (Gmax - Gmin)\n";
    }
    else if (   (!settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymmetryFunction::ST_CENTER;
        log << strpr("Scaling type::ST_CENTER (%d)\n", scalingType);
        log << "Gs = G - Gmean\n";
    }
    else if (   ( settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymmetryFunction::ST_SCALECENTER;
        log << strpr("Scaling type::ST_SCALECENTER (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)\n";
    }
    else if (settings.keywordExists("scale_symmetry_functions_sigma"))
    {
        scalingType = SymmetryFunction::ST_SCALESIGMA;
        log << strpr("Scaling type::ST_SCALESIGMA (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / Gsigma\n";
    }
    else
    {
        scalingType = SymmetryFunction::ST_NONE;
        log << strpr("Scaling type::ST_NONE (%d)\n", scalingType);
        log << "Gs = G\n";
        log << "WARNING: No symmetry function scaling!\n";
    }

    double Smin = 0.0;
    double Smax = 0.0;
    if (scalingType == SymmetryFunction::ST_SCALE ||
        scalingType == SymmetryFunction::ST_SCALECENTER ||
        scalingType == SymmetryFunction::ST_SCALESIGMA)
    {
        if (settings.keywordExists("scale_min_short"))
        {
            Smin = atof(settings["scale_min_short"].c_str());
        }
        else
        {
            log << "WARNING: Keyword \"scale_min_short\" not found.\n";
            log << "         Default value for Smin = 0.0.\n";
            Smin = 0.0;
        }

        if (settings.keywordExists("scale_max_short"))
        {
            Smax = atof(settings["scale_max_short"].c_str());
        }
        else
        {
            log << "WARNING: Keyword \"scale_max_short\" not found.\n";
            log << "         Default value for Smax = 1.0.\n";
            Smax = 1.0;
        }

        log << strpr("Smin = %f\n", Smin);
        log << strpr("Smax = %f\n", Smax);
    }

    log << strpr("Symmetry function scaling statistics from file: %s\n",
                 fileName.c_str());
    log << "-----------------------------------------"
           "--------------------------------------\n";
    ifstream file;
    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    string line;
    vector<string> lines;
    while (getline(file, line))
    {
        if (line.at(0) != '#') lines.push_back(line);
    }
    file.close();

    log << "\n";
    log << "Abbreviations:\n";
    log << "--------------\n";
    log << "ind ..... Symmetry function index.\n";
    log << "min ..... Minimum symmetry function value.\n";
    log << "max ..... Maximum symmetry function value.\n";
    log << "mean .... Mean symmetry function value.\n";
    log << "sigma ... Standard deviation of symmetry function values.\n";
    log << "sf ...... Scaling factor for derivatives.\n";
    log << "Smin .... Desired minimum scaled symmetry function value.\n";
    log << "Smax .... Desired maximum scaled symmetry function value.\n";
    log << "t ....... Scaling type.\n";
    log << "\n";
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->setScaling(scalingType, lines, Smin, Smax);
        log << strpr("Scaling data for symmetry functions element %2s :\n",
                     it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind       min       max      mean     sigma        sf  Smin  Smax t\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionScaling();
        log << "-----------------------------------------"
               "--------------------------------------\n";
        lines.erase(lines.begin(), lines.begin() + it->numSymmetryFunctions());
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionGroups()
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION GROUPS *****"
           "**************************************\n";
    log << "\n";

    log << "Abbreviations:\n";
    log << "--------------\n";
    log << "ind .... Symmetry function group index.\n";
    log << "ec ..... Central atom element.\n";
    log << "ty ..... Symmetry function type.\n";
    log << "e1 ..... Neighbor 1 element.\n";
    log << "e2 ..... Neighbor 2 element.\n";
    log << "eta .... Gaussian width eta.\n";
    log << "rs ..... Shift distance of Gaussian.\n";
    log << "la ..... Angle prefactor lambda.\n";
    log << "zeta ... Angle term exponent zeta.\n";
    log << "rc ..... Cutoff radius.\n";
    log << "ct ..... Cutoff type.\n";
    log << "ca ..... Cutoff alpha.\n";
    log << "ln ..... Line number in settings file.\n";
    log << "mi ..... Member index.\n";
    log << "sfi .... Symmetry function index.\n";
    log << "e ...... Recalculate exponential term.\n";
    log << "\n";
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->setupSymmetryFunctionGroups();
        log << strpr("Short range atomic symmetry function groups "
                     "element %2s :\n", it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind ec ty e1 e2       eta        rs la "
               "zeta        rc ct   ca    ln   mi  sfi e\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionGroups();
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionStatistics(bool collectStatistics,
                                           bool collectExtrapolationWarnings,
                                           bool writeExtrapolationWarnings,
                                           bool stopOnExtrapolationWarnings)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION STATISTICS *"
           "**************************************\n";
    log << "\n";

    log << "Equal symmetry function statistics for all elements.\n";
    log << strpr("Collect min/max/mean/sigma                        : %d\n",
                 (int)collectStatistics);
    log << strpr("Collect extrapolation warnings                    : %d\n",
                 (int)collectExtrapolationWarnings);
    log << strpr("Write extrapolation warnings immediately to stderr: %d\n",
                 (int)writeExtrapolationWarnings);
    log << strpr("Halt on any extrapolation warning                 : %d\n",
                 (int)stopOnExtrapolationWarnings);
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.collectStatistics = collectStatistics;
        it->statistics.collectExtrapolationWarnings =
                                                  collectExtrapolationWarnings;
        it->statistics.writeExtrapolationWarnings = writeExtrapolationWarnings;
        it->statistics.stopOnExtrapolationWarnings =
                                                   stopOnExtrapolationWarnings;
    }

    checkExtrapolationWarnings = collectStatistics
                              || collectExtrapolationWarnings
                              || writeExtrapolationWarnings
                              || stopOnExtrapolationWarnings;

    log << "*****************************************"
           "**************************************\n";
    return;
}

void Mode::setupNeuralNetwork()
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORKS **************"
           "**************************************\n";
    log << "\n";

    int const numLayers = 2 +
                          atoi(settings["global_hidden_layers_short"].c_str());
    int* numNeuronsPerLayer = new int[numLayers];
    NeuralNetwork::ActivationFunction* activationFunctionsPerLayer =
        new NeuralNetwork::ActivationFunction[numLayers];
    vector<string> numNeuronsPerHiddenLayer =
        split(reduce(settings["global_nodes_short"]));
    vector<string> activationFunctions =
        split(reduce(settings["global_activation_short"]));

    for (int i = 0; i < numLayers; i++)
    {
        if (i == 0)
        {
            numNeuronsPerLayer[i] = 0;
            activationFunctionsPerLayer[i] = NeuralNetwork::AF_IDENTITY;
        }
        else if (i == numLayers - 1)
        {
            numNeuronsPerLayer[i] = 1;
            activationFunctionsPerLayer[i] = NeuralNetwork::AF_IDENTITY;
        }
        else
        {
            numNeuronsPerLayer[i] =
                atoi(numNeuronsPerHiddenLayer.at(i-1).c_str());
            if (activationFunctions.at(i-1) == "l")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_IDENTITY;
            }
            else if (activationFunctions.at(i-1) == "t")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_TANH;
            }
            else if (activationFunctions.at(i-1) == "s")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_LOGISTIC;
            }
            else if (activationFunctions.at(i-1) == "p")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_SOFTPLUS;
            }
            else if (activationFunctions.at(i-1) == "r")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_RELU;
            }
            else if (activationFunctions.at(i-1) == "g")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_GAUSSIAN;
            }
            else if (activationFunctions.at(i-1) == "c")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_COS;
            }
            else if (activationFunctions.at(i-1) == "S")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_REVLOGISTIC;
            }
            else if (activationFunctions.at(i-1) == "e")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_EXP;
            }
            else if (activationFunctions.at(i-1) == "h")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_HARMONIC;
            }
            else
            {
                throw runtime_error("ERROR: Unknown activation function.\n");
            }
        }

    }

    bool normalizeNeurons = settings.keywordExists("normalize_nodes");
    log << strpr("Normalize neurons (all elements): %d\n",
                 (int)normalizeNeurons);
    log << "-----------------------------------------"
           "--------------------------------------\n";

    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        numNeuronsPerLayer[0] = it->numSymmetryFunctions();
        it->neuralNetwork = new NeuralNetwork(numLayers,
                                              numNeuronsPerLayer,
                                              activationFunctionsPerLayer);
        it->neuralNetwork->setNormalizeNeurons(normalizeNeurons);
        log << strpr("Atomic short range NN for "
                     "element %2s :\n", it->getSymbol().c_str());
        log << it->neuralNetwork->info();
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }

    delete[] numNeuronsPerLayer;
    delete[] activationFunctionsPerLayer;

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupNeuralNetworkWeights(string const& fileNameFormat)
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORK WEIGHTS *******"
           "**************************************\n";
    log << "\n";

    log << strpr("Weight file name format: %s\n", fileNameFormat.c_str());
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        string fileName = strpr(fileNameFormat.c_str(), it->getAtomicNumber());
        log << strpr("Weight file for element %2s: %s\n",
                     it->getSymbol().c_str(),
                     fileName.c_str());
        ifstream file;
        file.open(fileName.c_str());
        if (!file.is_open())
        {
            throw runtime_error("ERROR: Could not open file: \"" + fileName
                                + "\".\n");
        }
        string line;
        vector<double> weights;
        while (getline(file, line))
        {
            if (line.at(0) != '#')
            {
                vector<string> splitLine = split(reduce(line));
                weights.push_back(atof(splitLine.at(0).c_str()));
            }
        }
        it->neuralNetwork->setConnections(&(weights.front()));
        file.close();
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::calculateSymmetryFunctions(Structure& structure,
                                      bool const derivatives)
{
    // Skip calculation for whole structure if results are already saved.
    if (structure.hasSymmetryFunctionDerivatives) return;
    if (structure.hasSymmetryFunctions && !derivatives) return;

    Atom* a = NULL;
    Element* e = NULL;
#ifdef _OPENMP
    #pragma omp parallel for private (a, e)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Pointer to atom.
        a = &(structure.atoms.at(i));

        // Skip calculation for individual atom if results are already saved.
        if (a->hasSymmetryFunctionDerivatives) continue;
        if (a->hasSymmetryFunctions && !derivatives) continue;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        a->numSymmetryFunctions = e->numSymmetryFunctions();

#ifndef NONEIGHCHECK
        // Check if atom has low number of neighbors.
        size_t numNeighbors = a->getNumNeighbors(
                                            minCutoffRadius.at(e->getIndex()));
        if (numNeighbors < minNeighbors.at(e->getIndex()))
        {
            log << strpr("WARNING: Structure %6zu Atom %6zu : %zu "
                         "neighbors.\n",
                         a->indexStructure,
                         a->index,
                         numNeighbors);
        }
#endif

        // Allocate symmetry function data vectors in atom.
        a->allocate(derivatives);

        // Calculate symmetry functions (and derivatives).
        e->calculateSymmetryFunctions(*a, derivatives);

        // Remember that symmetry functions of this atom have been calculated.
        a->hasSymmetryFunctions = true;
        if (derivatives) a->hasSymmetryFunctionDerivatives = true;
    }

    // If requested, check extrapolation warnings or update statistics.
    // Needed to shift this out of the loop above to make it thread-safe.
    if (checkExtrapolationWarnings)
    {
        for (size_t i = 0; i < structure.atoms.size(); ++i)
        {
            a = &(structure.atoms.at(i));
            e = &(elements.at(a->element));
            e->updateSymmetryFunctionStatistics(*a);
        }
    }

    // Remember that symmetry functions of this structure have been calculated.
    structure.hasSymmetryFunctions = true;
    if (derivatives) structure.hasSymmetryFunctionDerivatives = true;

    return;
}

void Mode::calculateSymmetryFunctionGroups(Structure& structure,
                                           bool const derivatives)
{
    // Skip calculation for whole structure if results are already saved.
    if (structure.hasSymmetryFunctionDerivatives) return;
    if (structure.hasSymmetryFunctions && !derivatives) return;

    Atom* a = NULL;
    Element* e = NULL;
#ifdef _OPENMP
    #pragma omp parallel for private (a, e)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Pointer to atom.
        a = &(structure.atoms.at(i));

        // Skip calculation for individual atom if results are already saved.
        if (a->hasSymmetryFunctionDerivatives) continue;
        if (a->hasSymmetryFunctions && !derivatives) continue;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        a->numSymmetryFunctions = e->numSymmetryFunctions();

#ifndef NONEIGHCHECK
        // Check if atom has low number of neighbors.
        size_t numNeighbors = a->getNumNeighbors(
                                            minCutoffRadius.at(e->getIndex()));
        if (numNeighbors < minNeighbors.at(e->getIndex()))
        {
            log << strpr("WARNING: Structure %6zu Atom %6zu : %zu "
                         "neighbors.\n",
                         a->indexStructure,
                         a->index,
                         numNeighbors);
        }
#endif

        // Allocate symmetry function data vectors in atom.
        a->allocate(derivatives);

        // Calculate symmetry functions (and derivatives).
        e->calculateSymmetryFunctionGroups(*a, derivatives);

        // Remember that symmetry functions of this atom have been calculated.
        a->hasSymmetryFunctions = true;
        if (derivatives) a->hasSymmetryFunctionDerivatives = true;
    }

    // If requested, check extrapolation warnings or update statistics.
    // Needed to shift this out of the loop above to make it thread-safe.
    if (checkExtrapolationWarnings)
    {
        for (size_t i = 0; i < structure.atoms.size(); ++i)
        {
            a = &(structure.atoms.at(i));
            e = &(elements.at(a->element));
            e->updateSymmetryFunctionStatistics(*a);
        }
    }

    // Remember that symmetry functions of this structure have been calculated.
    structure.hasSymmetryFunctions = true;
    if (derivatives) structure.hasSymmetryFunctionDerivatives = true;

    return;
}

void Mode::calculateAtomicNeuralNetworks(Structure& structure,
                                         bool const derivatives) const
{
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        Element const& e = elements.at(it->element);
        e.neuralNetwork->setInput(&((it->G).front()));
        e.neuralNetwork->propagate();
        if (derivatives) e.neuralNetwork->calculateDEdG(&((it->dEdG).front()));
        e.neuralNetwork->getOutput(&(it->energy));
    }

    return;
}
void Mode::calculateAtomicChargesNeuralNetworks(Structure& structure, bool const derivatives) const
{
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        Element const& e = elements.at(it->element);
        e.neuralNetwork->setInput(&((it->G).front()));
        e.neuralNetwork->propagate();
        if (derivatives) e.neuralNetwork->calculateDEdG(&((it->dEdG).front()));
        e.neuralNetwork->getOutput(&(it->charge));
    }
    return;
}

void Mode::calculateEnergy(Structure& structure) const
{
    // Loop over all atoms and add atomic contributions to total energy.
    structure.energy = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        structure.energy += it->energy;
    }
    dftd3.add(structure, elements, true, false);
    return;
}

void Mode::calculateForces(Structure& structure) const
{
    Atom* ai = NULL;
    // Loop over all atoms, center atom i (ai).
#ifdef _OPENMP
    #pragma omp parallel for private(ai)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Set pointer to atom.
        ai = &(structure.atoms.at(i));

        // Reset forces.
        ai->f[0] = 0.0;
        ai->f[1] = 0.0;
        ai->f[2] = 0.0;

        // First add force contributions from atom i itself (gradient of
        // atomic energy E_i).
        for (size_t j = 0; j < ai->numSymmetryFunctions; ++j)
        {
            ai->f -= ai->dEdG.at(j) * ai->dGdr.at(j);
        }

        // Now loop over all neighbor atoms j of atom i. These may hold
        // non-zero derivatives of their symmetry functions with respect to
        // atom i's coordinates. Some atoms may appear multiple times in the
        // neighbor list because of periodic boundary conditions. To avoid
        // that the same contributions are added multiple times use the
        // "unique neighbor" list (but skip the first entry, this is always
        // atom i itself).
        for (vector<size_t>::const_iterator it =
             ai->neighborsUnique.begin() + 1;
             it != ai->neighborsUnique.end(); ++it)
        {
            // Define shortcut for atom j (aj).
            Atom& aj = structure.atoms.at(*it);

            // Loop over atom j's neighbors (n), atom i should be one of them.
            for (vector<Atom::Neighbor>::const_iterator n =
                 aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
            {
                // If atom j's neighbor is atom i add force contributions.
                if (n->index == ai->index)
                {
                    for (size_t j = 0; j < aj.numSymmetryFunctions; ++j)
                    {
                        ai->f -= aj.dEdG.at(j) * n->dGdr.at(j);
                    }
                }
            }
        }
    }
    dftd3.add(structure, elements, false, true);

    return;
}
void Mode::calculateEnergyAndForces(Structure& structure) const
{
    structure.energy = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        structure.energy += it->energy;
    }
    Atom* ai = NULL;
    // Loop over all atoms, center atom i (ai).
#ifdef _OPENMP
    #pragma omp parallel for private(ai)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Set pointer to atom.
        ai = &(structure.atoms.at(i));

        // Reset forces.
        ai->f[0] = 0.0;
        ai->f[1] = 0.0;
        ai->f[2] = 0.0;

        // First add force contributions from atom i itself (gradient of
        // atomic energy E_i).
        for (size_t j = 0; j < ai->numSymmetryFunctions; ++j)
        {
            ai->f -= ai->dEdG.at(j) * ai->dGdr.at(j);
        }

        // Now loop over all neighbor atoms j of atom i. These may hold
        // non-zero derivatives of their symmetry functions with respect to
        // atom i's coordinates. Some atoms may appear multiple times in the
        // neighbor list because of periodic boundary conditions. To avoid
        // that the same contributions are added multiple times use the
        // "unique neighbor" list (but skip the first entry, this is always
        // atom i itself).
        for (vector<size_t>::const_iterator it =
             ai->neighborsUnique.begin() + 1;
             it != ai->neighborsUnique.end(); ++it)
        {
            // Define shortcut for atom j (aj).
            Atom& aj = structure.atoms.at(*it);

            // Loop over atom j's neighbors (n), atom i should be one of them.
            for (vector<Atom::Neighbor>::const_iterator n =
                 aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
            {
                // If atom j's neighbor is atom i add force contributions.
                if (n->index == ai->index)
                {
                    for (size_t j = 0; j < aj.numSymmetryFunctions; ++j)
                    {
                        ai->f -= aj.dEdG.at(j) * n->dGdr.at(j);
                    }
                }
            }
        }
    }
    dftd3.add(structure, elements, true, true);

    return;
}
void Mode::calculateCharge(Structure& structure) const
{
    // Loop over all atoms and add atomic contributions to total charge.
    structure.charge = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        structure.charge += it->charge;
    }

    return;
}
void Mode::calculateDipole(Structure& structure) const
{
    // Loop over all atoms and add atomic contributions to dipole.
    for(int i=0;i<3;i++) structure.dipole[i] = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
    	for(int i=0;i<3;i++) structure.dipole[i] += it->charge*it->r[i];
    }

    return;
}
double** Mode::computedDipole(Structure& structure) const
{
    // Loop over all atoms and add atomic contributions to dipole.
	size_t nAtoms = structure.atoms.size();
	size_t nAtoms3 = 3*nAtoms;
	double** dmu = new double*[3];
	for(size_t k=0;k<3;k++) dmu[k] = new double [nAtoms3];// 3*nAtoms coordinates and mux, muy, muz for the molecule
	for(size_t k=0;k<3;k++) 
	for(size_t i=0;i<nAtoms3;i++) dmu[k][i] = 0.0;

	// mux = sum Qi xi
	// muy = sum Qi yi
	// muz = sum Qi zi
	// Loop over all atoms, center atom i (ai).
//#ifdef _OPENMP
//    #pragma omp parallel 
//#endif
	for (size_t i = 0; i < structure.atoms.size(); ++i)
	{
        	// Set pointer to atom.
		Atom* ai = &(structure.atoms.at(i));
		for(size_t k=0;k<3;k++) dmu[k][3*i+k] += ai->charge;

		// First add contributions from atom i itself (gradient of atomic charge Q_i).
        	for (size_t j = 0; j < ai->numSymmetryFunctions; ++j)
        	{
			for(size_t l=0;l<3;l++) 
			for(size_t k=0;k<3;k++) 
				dmu[k][3*i+l] += ai->dEdG.at(j) * ai->dGdr.at(j)[l]*ai->r[k];
        	}

		// Now loop over all neighbor atoms j of atom i. These may hold
		// non-zero derivatives of their symmetry functions with respect to
        	// atom i's coordinates. Some atoms may appear multiple times in the
        	// neighbor list because of periodic boundary conditions. To avoid
        	// that the same contributions are added multiple times use the
        	// "unique neighbor" list (but skip the first entry, this is always
        	// atom i itself).
        	for (vector<size_t>::const_iterator it = ai->neighborsUnique.begin()+1; it != ai->neighborsUnique.end(); ++it)
        	{
            		// Define shortcut for atom j (aj).
            		Atom& aj = structure.atoms.at(*it);

            		// Loop over atom j's neighbors (n), atom i should be one of them.
            		for (vector<Atom::Neighbor>::const_iterator n = aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
            		{
                		// If atom j's neighbor is atom i add contributions.
                		if (n->index == ai->index)
                		{
                    			for (size_t j = 0; j < aj.numSymmetryFunctions; ++j)
                    			{
						for(size_t l=0;l<3;l++) 
							for(size_t k=0;k<3;k++) 
								dmu[k][3*i+l] += aj.dEdG.at(j) * n->dGdr.at(j)[l]*aj.r[k];
                    			}
                		}
            		}
        	}
	}
	return dmu;
}
#ifdef HIGH_DERIVATIVES
// Derivatives of Mu = sum_(i=1,nAtoms) ri Qi
Derivatives* Mode::computeHighDerivativesDipoleMem(Structure& structure, const int order) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHighDerivatives\n");
              return NULL;
	}
	Derivatives dnEdGn(order, maxnumSymFunc);

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;

	Derivatives* deriv = new Derivatives[3];
	for(int c=0;c<3;c++) 
	{
		deriv[c] = Derivatives(order,nAtoms3);
		deriv[c].reset();
	}

	Derivatives Q(order,nAtoms3);
	Derivatives r(order,nAtoms3);
	int index[order];

	Derivatives G[order];
	for( int i = 0; i<order; i++) G[i] = Derivatives(order,nAtoms3);

	Derivatives GAll[maxnumSymFunc];
	for(size_t k = 0; k<maxnumSymFunc; k++)
		GAll[k] = Derivatives(order,nAtoms3);

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
		dnEdGn.compute(e.neuralNetwork);

		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
			e.getSymmetryFunction(k).computeDerivatives(*ita, GAll[k]);

		// compute charge derivatives
		Q.reset();
		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			index[0] = k;
			G[0] = GAll[k];
			Q.addTodF(dnEdGn, G, index, 1);// add only First derivatives contributions
			if(order<2) continue;
			for(size_t l = 0; l<=k; l++)
			{
				index[1] = l;
				G[1] = GAll[l];
				Q.addTodF(dnEdGn, G, index, 2);// add only second derivatives contributions
				if(order<3) continue;
				for(size_t m = 0; m<=l; m++)
				{
					index[2] = m;
					G[2] = GAll[m];
					Q.addTodF(dnEdGn, G, index, 3);// add only thirds derivatives contributions
					if(order<4) continue;
					for(size_t n = 0; n<=m; n++)
					{
						index[3] = n;
						G[3] = GAll[n];
						Q.addTodF(dnEdGn, G, index, 4);// add only fourth derivatives contributions
					}
				}
    			}
		}
		for(int c=0;c<3;c++) 
		{
			r.reset();
			r.setValue(ita->r[c]);
			r.setdFValue(1.0, 3*ita->index+c);
			deriv[c] += r*Q;
		}
	}
	return deriv;
}
// Derivatives of Mu = sum_(i=1,nAtoms) ri Qi
Derivatives* Mode::computeHighDerivativesDipoleFile(Structure& structure, const int order) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHighDerivatives\n");
              return NULL;
	}
	Derivatives dnEdGn(order, maxnumSymFunc);

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;

	Derivatives* deriv = new Derivatives[3];
	for(int c=0;c<3;c++) 
	{
		deriv[c] = Derivatives(order,nAtoms3);
		deriv[c].reset();
	}

	Derivatives Q(order,nAtoms3);
	Derivatives r(order,nAtoms3);
	int index[order];

	Derivatives G[order];
	for( int i = 0; i<order; i++) G[i] = Derivatives(order,nAtoms3);

	string fileNames[maxnumSymFunc];
	for(size_t k = 0; k<maxnumSymFunc; k++)
	{
		stringstream oss;
		oss<<"G"<<k<<".bin";
		oss>>fileNames[k];
	}


	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
		dnEdGn.compute(e.neuralNetwork);
		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			e.getSymmetryFunction(k).computeDerivatives(*ita, G[0]);
			G[0].save(fileNames[k].c_str());
		}

		// compute charge derivatives
		Q.reset();
		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			index[0] = k;
			G[0].read(fileNames[k].c_str());
			Q.addTodF(dnEdGn, G, index, 1);// add only First derivatives contributions
			if(order<2) continue;
			for(size_t l = 0; l<=k; l++)
			{
				index[1] = l;
				if(k==l) G[1] = G[0];
				else G[1].read(fileNames[l].c_str());
				Q.addTodF(dnEdGn, G, index, 2);// add only second derivatives contributions
				if(order<3) continue;
				for(size_t m = 0; m<=l; m++)
				{
					index[2] = m;
					if(m==k) G[2] = G[0];
					else if(m==l) G[2] = G[1];
					else G[2].read(fileNames[m].c_str());
					Q.addTodF(dnEdGn, G, index, 3);// add only thirds derivatives contributions
					if(order<4) continue;
					for(size_t n = 0; n<=m; n++)
					{
						index[3] = n;
						if(n==k) G[3] = G[0];
						else if(n==l) G[3] = G[1];
						else if(n==m) G[3] = G[2];
						else G[3].read(fileNames[n].c_str());
						Q.addTodF(dnEdGn, G, index, 4);// add only fourth derivatives contributions
					}
				}
    			}
		}
		for(int c=0;c<3;c++) 
		{
			r.reset();
			r.setValue(ita->r[c]);
			r.setdFValue(1.0, 3*ita->index+c);
			deriv[c] += r*Q;
		}
	}
	return deriv;
}
// Derivatives of Mu = sum_(i=1,nAtoms) ri Qi
Derivatives* Mode::computeHighDerivativesDipole(Structure& structure, const int order) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHighDerivatives\n");
              return NULL;
	}
	Derivatives dnEdGn(order, maxnumSymFunc);

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;

	Derivatives* deriv = new Derivatives[3];
	for(int c=0;c<3;c++) 
	{
		deriv[c] = Derivatives(order,nAtoms3);
		deriv[c].reset();
	}

	Derivatives Q(order,nAtoms3);
	Derivatives r(order,nAtoms3);
	int index[order];

	Derivatives G[order];
	for( int i = 0; i<order; i++) G[i] = Derivatives(order,nAtoms3);

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
		dnEdGn.compute(e.neuralNetwork);
		// compute charge derivatives
		Q.reset();
		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			index[0] = k;
			e.getSymmetryFunction(k).computeDerivatives(*ita, G[0]);
			Q.addTodF(dnEdGn, G, index, 1);// add first second derivatives contributions
			if(order<2) continue;
			for(size_t l = 0; l<=k; l++)
			{
				index[1] = l;
				if(k==l) G[1] = G[0];
				else e.getSymmetryFunction(l).computeDerivatives(*ita, G[1]);
				Q.addTodF(dnEdGn, G, index, 2);// add only second derivatives contributions
				if(order<3) continue;
				for(size_t m = 0; m<=l; m++)
				{
					index[2] = m;
					if(m==k) G[2] = G[0];
					else if(m==l) G[2] = G[1];
					else e.getSymmetryFunction(m).computeDerivatives(*ita, G[2]);
					Q.addTodF(dnEdGn, G, index, 3);// add only thirds derivatives contributions
					if(order<4) continue;
					for(size_t n = 0; n<=m; n++)
					{
						index[3] = n;
						if(n==k) G[3] = G[0];
						else if(n==l) G[3] = G[1];
						else if(n==m) G[3] = G[2];
						else e.getSymmetryFunction(n).computeDerivatives(*ita, G[3]);
						Q.addTodF(dnEdGn, G, index, 4);// add only fourth derivatives contributions
					}
				}
    			}
		}
		for(int c=0;c<3;c++) 
		{
			r.reset();
			r.setValue(ita->r[c]);
			r.setdFValue(1.0, 3*ita->index+c);
			deriv[c] += r*Q;
		}
	}
	return deriv;
}
Derivatives* Mode::computeHighDerivativesDipole(Structure& structure, const int order, int method) const
{
	if(method==0)
		return computeHighDerivativesDipoleMem(structure, order);
	else if(method==1)
		return computeHighDerivativesDipoleFile(structure, order);
	else
		return computeHighDerivativesDipole(structure, order);
}
Derivatives Mode::computeHighDerivativesFile(Structure& structure, const int order) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHighDerivativesFile\n");
              return Derivatives();
	}
	Derivatives dnEdGn(order, maxnumSymFunc);

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;

	Derivatives deriv(order,nAtoms3);
	deriv.reset();
	int index[order];

	Derivatives G[order];
	for( int i = 0; i<order; i++) G[i] = Derivatives(order,nAtoms3);

	string fileNames[maxnumSymFunc];
	for(size_t k = 0; k<maxnumSymFunc; k++)
	{
		stringstream oss;
		oss<<"G"<<k<<".bin";
		oss>>fileNames[k];
	}

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
		dnEdGn.compute(e.neuralNetwork);
		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			e.getSymmetryFunction(k).computeDerivatives(*ita, G[0]);
			G[0].save(fileNames[k].c_str());
		}

		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			index[0] = k;
			G[0].read(fileNames[k].c_str());
			deriv.addTodF(dnEdGn, G, index, 1);// add only First derivatives contributions
			if(order<2) continue;
			for(size_t l = 0; l<=k; l++)
			{
				index[1] = l;
				if(k==l) G[1] = G[0];
				else G[1].read(fileNames[l].c_str());
				deriv.addTodF(dnEdGn, G, index, 2);// add only second derivatives contributions
				if(order<3) continue;
				for(size_t m = 0; m<=l; m++)
				{
					index[2] = m;
					if(m==k) G[2] = G[0];
					else if(m==l) G[2] = G[1];
					else G[2].read(fileNames[m].c_str());
					deriv.addTodF(dnEdGn, G, index, 3);// add only thirds derivatives contributions
					if(order<4) continue;
					for(size_t n = 0; n<=m; n++)
					{
						index[3] = n;
						if(n==k) G[3] = G[0];
						else if(n==l) G[3] = G[1];
						else if(n==m) G[3] = G[2];
						else G[3].read(fileNames[n].c_str());
						deriv.addTodF(dnEdGn, G, index, 4);// add only fourth derivatives contributions
					}
				}

    			}
		}
	}
	for(size_t k = 0; k<maxnumSymFunc; k++)
		remove(fileNames[k].c_str());
	return deriv;
}
Derivatives Mode::computeHighDerivativesMem(Structure& structure, const int order) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHighDerivatives\n");
              return Derivatives();
	}
	Derivatives dnEdGn(order, maxnumSymFunc);

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;

	Derivatives deriv(order,nAtoms3);
	deriv.reset();
	int index[order];

	Derivatives G[order];
	for( int i = 0; i<order; i++) G[i] = Derivatives(order,nAtoms3);

	Derivatives GAll[maxnumSymFunc];
	for(size_t k = 0; k<maxnumSymFunc; k++)
		GAll[k] = Derivatives(order,nAtoms3);

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
		dnEdGn.compute(e.neuralNetwork);
		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
			e.getSymmetryFunction(k).computeDerivatives(*ita, GAll[k]);

		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			index[0] = k;
			G[0] = GAll[k];
			deriv.addTodF(dnEdGn, G, index, 1);// add only First derivatives contributions
			if(order<2) continue;
			for(size_t l = 0; l<=k; l++)
			{
				index[1] = l;
				G[1] = GAll[l];
				deriv.addTodF(dnEdGn, G, index, 2);// add only second derivatives contributions
				if(order<3) continue;
				for(size_t m = 0; m<=l; m++)
				{
					index[2] = m;
					G[2] = GAll[m];
					deriv.addTodF(dnEdGn, G, index, 3);// add only thirds derivatives contributions
					if(order<4) continue;
					for(size_t n = 0; n<=m; n++)
					{
						index[3] = n;
						G[3] = GAll[n];
						deriv.addTodF(dnEdGn, G, index, 4);// add only fourth derivatives contributions
					}
				}

    			}
		}
	}
	return deriv;
}
Derivatives Mode::computeHighDerivatives(Structure& structure, const int order) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHighDerivatives\n");
              return Derivatives();
	}
	Derivatives dnEdGn(order, maxnumSymFunc);

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;

	Derivatives deriv(order,nAtoms3);
	deriv.reset();
	int index[order];

	Derivatives G[order];
	for( int i = 0; i<order; i++) G[i] = Derivatives(order,nAtoms3);

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
		dnEdGn.compute(e.neuralNetwork);

		for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
		{
			index[0] = k;
			e.getSymmetryFunction(k).computeDerivatives(*ita, G[0]);
			deriv.addTodF(dnEdGn, G, index, 1);// add only First derivatives contributions
			if(order<2) continue;
			for(size_t l = 0; l<=k; l++)
			{
				index[1] = l;
				if(k==l) G[1] = G[0];
				else e.getSymmetryFunction(l).computeDerivatives(*ita, G[1]);
				deriv.addTodF(dnEdGn, G, index, 2);// add only second derivatives contributions
				if(order<3) continue;
				for(size_t m = 0; m<=l; m++)
				{
					index[2] = m;
					if(m==k) G[2] = G[0];
					else if(m==l) G[2] = G[1];
					else e.getSymmetryFunction(m).computeDerivatives(*ita, G[2]);
					deriv.addTodF(dnEdGn, G, index, 3);// add only thirds derivatives contributions
					if(order<4) continue;
					for(size_t n = 0; n<=m; n++)
					{
						index[3] = n;
						if(n==k) G[3] = G[0];
						else if(n==l) G[3] = G[1];
						else if(n==m) G[3] = G[2];
						else e.getSymmetryFunction(n).computeDerivatives(*ita, G[3]);
						deriv.addTodF(dnEdGn, G, index, 4);// add only fourth derivatives contributions
					}
				}
    			}
		}
	}
	return deriv;
}
Derivatives Mode::computeHighDerivatives(Structure& structure, const int order, int method) const
{
	if(method==0)
		return computeHighDerivativesMem(structure, order);
	else if(method==1)
		return computeHighDerivativesFile(structure, order);
	else
		return computeHighDerivatives(structure, order);
}
Derivatives Mode::computeFourthFile(Structure& structure) const
{
	return computeHighDerivatives(structure, 4, 1);
}
Derivatives Mode::computeFourthMem(Structure& structure) const
{
	return computeHighDerivatives(structure, 4, 0);
}
Derivatives Mode::computeFourth(Structure& structure) const
{
	return computeHighDerivatives(structure, 4, 2);
}
Derivatives Mode::computeThirdFile(Structure& structure) const
{
	return computeHighDerivatives(structure, 3, 1);
}
Derivatives Mode::computeThirdMem(Structure& structure) const
{
	return computeHighDerivatives(structure, 3, 0);
}
Derivatives Mode::computeThird(Structure& structure) const
{
	return computeHighDerivatives(structure, 3, 2);
}
// Implemented to test Derivatives class. 
// For Hessian, use the next method: computeHessian(Structure& structure, size_t& derivSize)
// it is more faster 
Derivatives Mode::computeHessian(Structure& structure) const
{
	return computeHighDerivatives(structure, 2, 0);
}
double* Mode::computeHessian(Structure& structure, size_t& derivSize) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHessian\n");
              return NULL;
	}
	double** d2EdG2 = new double*[maxnumSymFunc];
	for(size_t i=0;i<maxnumSymFunc;i++) 
		d2EdG2[i] = new double[i+1];

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;
	vector<double> v(nAtoms3);
	derivSize = nAtoms3*(nAtoms3+1)/2;
	double* deriv = new double [derivSize];
	vector< vector<double> > derivGij(nAtoms3,v);

    	for (size_t i = 0; i < derivSize; ++i) deriv[i] = 0.0;

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
        	//e.neuralNetwork->calculateDEdG(&((ita->dEdG).front()));
        	e.neuralNetwork->calculatedEdG(&((ita->dEdG).front()));
        	e.neuralNetwork->calculateD2EdG2(d2EdG2);

		for(size_t j = 0; j<ita->numSymmetryFunctions; j++)
		{
			e.getSymmetryFunction(j).compute2Derivatives(*ita, derivGij);
			for(size_t k=0;k<nAtoms3;k++) 
			for(size_t l=0;l<=k;l++) 
        		{
                                int index = l + k*(k+1)/2;
            			deriv[index] += ita->dEdG.at(j) * derivGij[k][l];
        		}
    		}
        	// Now loop over all neighbor atoms j of atom i.
		//Some atoms may appear multiple times in the
        	// neighbor list because of periodic boundary conditions. To avoid
        	// that the same contributions are added multiple times use the
        	// "unique neighbor" list  (including i atom itself)
        	for (vector<size_t>::const_iterator itn1 = ita->neighborsUnique.begin(); itn1 != ita->neighborsUnique.end(); ++itn1)
        	for (vector<size_t>::const_iterator itn2 = ita->neighborsUnique.begin(); itn2 != ita->neighborsUnique.end(); ++itn2)
        	{
			size_t in1 = *itn1;
			vector<Atom::Neighbor>::iterator n1 = ita->neighbors.begin();
			for (vector<Atom::Neighbor>::iterator n = ita->neighbors.begin(); n != ita->neighbors.end(); ++n) if(n->index == in1) n1 = n;
			size_t in2 = *itn2;
			vector<Atom::Neighbor>::iterator n2 = ita->neighbors.begin();
			for (vector<Atom::Neighbor>::iterator n = ita->neighbors.begin(); n != ita->neighbors.end(); ++n) if(n->index == in2) n2 = n;


			std::vector<Vec3D>*  dGdr1 = &ita->dGdr;
			if(in1==n1->index) dGdr1=&n1->dGdr;
			std::vector<Vec3D>*  dGdr2 = &ita->dGdr;
			if(in2==n2->index) dGdr2=&n2->dGdr;

			for(size_t j = 0; j<ita->numSymmetryFunctions; j++)
			for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
				for(size_t ix=0;ix<3;ix++) 
				for(size_t jx=0;jx<3;jx++) 
        			{	
					int jj=3*in1+ix;
					int kk=3*in2+jx;
					if(jj>kk) continue;
                                	int index = jj + kk*(kk+1)/2;
            				double d = (j>k)? d2EdG2[j][k]:d2EdG2[k][j];
            				deriv[index] += d*dGdr1->at(j)[ix]*dGdr2->at(k)[jx];
        			}
                }
	}
	for(size_t i=0;i<maxnumSymFunc;i++) if(d2EdG2[i]) delete[] d2EdG2[i];
	if(d2EdG2) delete[] d2EdG2;
	return deriv;
}
// d xi Qi/dadb = dxi/dadb Qi + dxi/da dQi/db + dxi/db dQi/da + xi d2Qi/dadb
//              =  0          + dxi/da dQi/db + dxi/db dQi/da + xi d2Qi/dadb
//              =               (xi==a)?1:0 dQi/db + (xi==b)?1:0 dQi/da + xi d2Qi/dadb
double** Mode::computed2Dipole(Structure& structure, size_t& derivSize) const
{
	size_t maxnumSymFunc = 0; 
    	for (vector<Atom>::iterator it = structure.atoms.begin(); it != structure.atoms.end(); ++it)
		if(maxnumSymFunc<it->numSymmetryFunctions) maxnumSymFunc = it->numSymmetryFunctions;

	if(maxnumSymFunc<1)
	{
	      throw runtime_error("maxnumSymFunc is null in computeHessian\n");
              return NULL;
	}
	double** d2EdG2 = new double*[maxnumSymFunc];
	for(size_t i=0;i<maxnumSymFunc;i++) 
		d2EdG2[i] = new double[i+1];

	size_t nAtoms = structure.numAtoms;
	size_t nAtoms3 = 3*nAtoms;
	vector<double> v(nAtoms3);
	derivSize = nAtoms3*(nAtoms3+1)/2;
	double** d2mu = new double* [3];
	for(int c=0;c<3;c++) d2mu[c] = new double [derivSize];
	vector< vector<double> > derivGij(nAtoms3,v);

	for(int c=0;c<3;c++) 
    	for (size_t i = 0; i < derivSize; ++i) d2mu[c][i] = 0.0;

	//size_t i=1;
	for (vector<Atom>::iterator ita = structure.atoms.begin(); ita != structure.atoms.end(); ++ita)
	{
        	Element const& e = elements.at(ita->element);
        	e.neuralNetwork->setInput(&((ita->G).front()));
        	e.neuralNetwork->propagate();
        	//e.neuralNetwork->calculateDEdG(&((ita->dEdG).front()));
        	e.neuralNetwork->calculatedEdG(&((ita->dEdG).front()));
        	e.neuralNetwork->calculateD2EdG2(d2EdG2);

		// contribution of xi d2Qi/dadb
		for(size_t j = 0; j<ita->numSymmetryFunctions; j++)
		{
			e.getSymmetryFunction(j).compute2Derivatives(*ita, derivGij);
			for(size_t k=0;k<nAtoms3;k++) 
			for(size_t l=0;l<=k;l++) 
        		{
                                int index = l + k*(k+1)/2;
				double v = ita->dEdG.at(j) * derivGij[k][l];
				
				for(int c=0;c<3;c++) d2mu[c][index] += v*ita->r[c];
        		}
    		}
        	// Now loop over all neighbor atoms j of atom i.
		//Some atoms may appear multiple times in the
        	// neighbor list because of periodic boundary conditions. To avoid
        	// that the same contributions are added multiple times use the
        	// "unique neighbor" list  (including i atom itself)
        	for (vector<size_t>::const_iterator itn1 = ita->neighborsUnique.begin(); itn1 != ita->neighborsUnique.end(); ++itn1)
        	for (vector<size_t>::const_iterator itn2 = ita->neighborsUnique.begin(); itn2 != ita->neighborsUnique.end(); ++itn2)
        	{
			size_t in1 = *itn1;
			vector<Atom::Neighbor>::iterator n1 = ita->neighbors.begin();
			for (vector<Atom::Neighbor>::iterator n = ita->neighbors.begin(); n != ita->neighbors.end(); ++n) if(n->index == in1) n1 = n;
			size_t in2 = *itn2;
			vector<Atom::Neighbor>::iterator n2 = ita->neighbors.begin();
			for (vector<Atom::Neighbor>::iterator n = ita->neighbors.begin(); n != ita->neighbors.end(); ++n) if(n->index == in2) n2 = n;



			std::vector<Vec3D>*  dGdr1 = &ita->dGdr;
			if(in1==n1->index) dGdr1=&n1->dGdr;
			std::vector<Vec3D>*  dGdr2 = &ita->dGdr;
			if(in2==n2->index) dGdr2=&n2->dGdr;

			for(size_t j = 0; j<ita->numSymmetryFunctions; j++)
			for(size_t k = 0; k<ita->numSymmetryFunctions; k++)
				for(size_t ix=0;ix<3;ix++) 
				for(size_t jx=0;jx<3;jx++) 
        			{	
					int jj=3*in1+ix;
					int kk=3*in2+jx;
					if(jj>kk) continue;
                                	int index = jj + kk*(kk+1)/2;
					double d = (j>k)?d2EdG2[j][k]:d2EdG2[k][j];
            				double v = d*dGdr1->at(j)[ix]*dGdr2->at(k)[jx];
					for(int c=0;c<3;c++) d2mu[c][index] += v*ita->r[c];
        			}
                }
		// contibution of (xi==a)?1:0 dQi/db + (xi==b)?1:0 dQi/da
        	for (size_t j = 0; j < ita->numSymmetryFunctions; ++j)
        	{
			for(int c=0;c<3;c++) 
			{
            			double v = ita->dEdG.at(j) * ita->dGdr.at(j)[c];
                                int l = 3*ita->index+c; 
				for(int c2=0;c2<=c;c2++) 
				{
            				double v2 = ita->dEdG.at(j) * ita->dGdr.at(j)[c2];
					int k = 3*ita->index+c2;
					int index = l + k*(k+1)/2;
					if(l>k) index = k + l*(l+1)/2;
					d2mu[c][index] += v2;
					d2mu[c2][index] += v;
				}
			}
        	}
        	// Now loop over all neighbor atoms j of atom i. 
        	for (vector<size_t>::const_iterator it = ita->neighborsUnique.begin() + 1; it != ita->neighborsUnique.end(); ++it)
        	{
            		// Define shortcut for atom j (aj).
            		Atom& aj = structure.atoms.at(*it);

            		// Loop over atom j's neighbors (n), atom i should be one of them.
            		for (vector<Atom::Neighbor>::const_iterator n = aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
            		{
                		// If atom j's neighbor is atom i add force contributions.
                		if (n->index == ita->index)
                		{
                    			for (size_t j = 0; j < aj.numSymmetryFunctions; ++j)
					for(int c=0;c<3;c++) 
					{
            					double v = aj.dEdG.at(j) * n->dGdr.at(j)[c];
                                		int l = 3*ita->index+c; 
						for(int c2=0;c2<3;c2++) 
						{
							int k = 3*aj.index+c2;
							int index = l + k*(k+1)/2;
							if(l>k) index = k + l*(l+1)/2;
							d2mu[c2][index] += v;
						}
					}
            			}
        		}
		}
	}
	for(size_t i=0;i<maxnumSymFunc;i++) if(d2EdG2[i]) delete[] d2EdG2[i];
	if(d2EdG2) delete[] d2EdG2;
	return d2mu;
}
#endif


void Mode::addEnergyOffset(Structure& structure, bool ref)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        if (ref)
        {
            structure.energyRef += structure.numAtomsPerElement.at(i)
                                 * elements.at(i).getAtomicEnergyOffset();
        }
        else
        {
            structure.energy += structure.numAtomsPerElement.at(i)
                              * elements.at(i).getAtomicEnergyOffset();
        }
    }

    return;
}

void Mode::removeEnergyOffset(Structure& structure, bool ref)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        if (ref)
        {
            structure.energyRef -= structure.numAtomsPerElement.at(i)
                                 * elements.at(i).getAtomicEnergyOffset();
        }
        else
        {
            structure.energy -= structure.numAtomsPerElement.at(i)
                              * elements.at(i).getAtomicEnergyOffset();
        }
    }

    return;
}

double Mode::getEnergyOffset(Structure const& structure) const
{
    double result = 0.0;

    for (size_t i = 0; i < numElements; ++i)
    {
        result += structure.numAtomsPerElement.at(i)
                * elements.at(i).getAtomicEnergyOffset();
    }

    return result;
}

double Mode::getEnergyWithOffset(Structure const& structure, bool ref) const
{
    double result;
    if (ref) result = structure.energyRef;
    else     result = structure.energy;

    for (size_t i = 0; i < numElements; ++i)
    {
        result += structure.numAtomsPerElement.at(i)
                * elements.at(i).getAtomicEnergyOffset();
    }

    return result;
}

double Mode::normalizedEnergy(double energy) const
{
    return energy * convEnergy; 
}

double Mode::normalizedEnergy(Structure const& structure, bool ref) const
{
    if (ref)
    {
        return (structure.energyRef - structure.numAtoms * meanEnergy)
               * convEnergy; 
    }
    else
    {
        return (structure.energy - structure.numAtoms * meanEnergy)
               * convEnergy; 
    }
}

double Mode::normalizedForce(double force) const
{
    return force * convEnergy / convLength;
}

double Mode::normalizedDipole(double dipole) const
{
    return dipole * convEnergy; 
}
Vec3D Mode::normalizedDipole(Structure const& structure, bool ref) const
{
    if (ref)
        return (structure.dipoleRef)* convDipole; 
    else
        return (structure.dipole)* convDipole; 
}
double Mode::normalizedDDipole(double ddipole) const
{
    return ddipole * convDipole / convLength;
}

double Mode::physicalEnergy(double energy) const
{
    return energy / convEnergy; 
}

double Mode::physicalEnergy(Structure const& structure, bool ref) const
{
    if (ref)
    {
        return structure.energyRef / convEnergy + structure.numAtoms
               * meanEnergy; 
    }
    else
    {
        return structure.energy / convEnergy + structure.numAtoms * meanEnergy; 
    }
}

double Mode::physicalForce(double force) const
{
    return force * convLength / convEnergy;
}
double Mode::physicalDipole(double dipole) const
{
    return dipole / convDipole; 
}
Vec3D Mode::physicalDipole(Structure const& structure, bool ref) const
{
    if (ref)
        return (structure.dipoleRef)/ convDipole; 
    else
        return (structure.dipole)/ convDipole; 
}
double Mode::physicalDDipole(double ddipole) const
{
    return ddipole / convDipole * convLength;
}

void Mode::convertToNormalizedUnits(Structure& structure) const
{
    structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convDipole);

    return;
}

void Mode::convertToPhysicalUnits(Structure& structure) const
{
    structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convDipole);

    return;
}

void Mode::resetExtrapolationWarnings()
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.resetExtrapolationWarnings();
    }

    return;
}

size_t Mode::getNumExtrapolationWarnings() const
{
    size_t numExtrapolationWarnings = 0;

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        numExtrapolationWarnings +=
            it->statistics.countExtrapolationWarnings();
    }

    return numExtrapolationWarnings;
}

vector<size_t> Mode::getNumSymmetryFunctions() const
{
    vector<size_t> v;

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        v.push_back(it->numSymmetryFunctions());
    }

    return v;
}

bool Mode::settingsKeywordExists(std::string const& keyword) const
{
    return settings.keywordExists(keyword);
}

string Mode::settingsGetValue(std::string const& keyword) const
{
    return settings.getValue(keyword);
}


void Mode::writePrunedSettingsFile(vector<size_t> prune, string fileName) const
{
    ofstream file(fileName.c_str());
    vector<string> settingsLines = settings.getSettingsLines();
    for (size_t i = 0; i < settingsLines.size(); ++i)
    {
        if (find(prune.begin(), prune.end(), i) != prune.end())
        {
            file << "# ";
        }
        file << settingsLines.at(i) << '\n';
    }
    file.close();

    return;
}

void Mode::writeSettingsFile(ofstream* const& file) const
{
    settings.writeSettingsFile(file);

    return;
}

vector<size_t> Mode::pruneSymmetryFunctionsRange(double threshold)
{
    vector<size_t> prune;

    // Check if symmetry functions have low range.
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        for (size_t i = 0; i < it->numSymmetryFunctions(); ++i)
        {
            SymmetryFunction const& s = it->getSymmetryFunction(i);
            if (fabs(s.getGmax() - s.getGmin()) < threshold)
            {
                prune.push_back(it->getSymmetryFunction(i).getLineNumber());
            }
        }
    }

    return prune;
}

vector<size_t> Mode::pruneSymmetryFunctionsSensitivity(
                                           double                  threshold,
                                           vector<vector<double> > sensitivity)
{
    vector<size_t> prune;

    for (size_t i = 0; i < numElements; ++i)
    {
        for (size_t j = 0; j < elements.at(i).numSymmetryFunctions(); ++j)
        {
            if (sensitivity.at(i).at(j) < threshold)
            {
                prune.push_back(
                        elements.at(i).getSymmetryFunction(j).getLineNumber());
            }
        }
    }

    return prune;
}
