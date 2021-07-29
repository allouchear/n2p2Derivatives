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

#include "Atom.h"
#include "PredictionES.h"
#include "Prediction.h"
#include "Structure.h"
#include "utility.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    if (argc != 1)
    {
        cout << "USAGE: " << argv[0] << "\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - inputES.nn (NNP settings for Electrostatic)\n"
             << "       - \"weightsES.%%03d.data\" (weights files for ElectroStatic)\n"
             << "       - input.nn (NNP settings for energies)\n"
             << "       - \"weights.%%03d.data\" (weights files for energies)\n";
        return 1;
    }


    ofstream logFile;
    logFile.open("nnp-predictES.log");
    PredictionES predictionES;
    Prediction prediction;
    predictionES.log.registerStreamPointer(&logFile);
    prediction.log.registerStreamPointer(&logFile);
    prediction.setup();
    predictionES.setup();
    prediction.log << "\n";
    prediction.log << "*** PREDICTION **************************"
                      "**************************************\n";
    prediction.log << "\n";
    prediction.log << "Reading structure file...\n";
    predictionES.readStructureFromFile("input.data");
    prediction.readStructureFromFile("input.data");
    Structure& s = prediction.structure;
    prediction.log << strpr("Structure contains %d atoms (%d elements).\n", s.numAtoms, s.numElements);
    prediction.log << "-------------------------------------------------------------------------------\n";
    prediction.predict();
    prediction.log << strpr("NNP energy: %16.8E\n", prediction.structure.energy);
    prediction.log << "NNP forces:\n";
    for (vector<Atom>::const_iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        prediction.log << strpr("%10zu %2s %16.8E %16.8E %16.8E\n",
                                it->index + 1,
                                prediction.elementMap[it->element].c_str(),
                                it->element,
                                it->f[0],
                                it->f[1],
                                it->f[2]);
    }
    prediction.log << "-------------------------------------------------------------------------------\n";
    predictionES.predict();
    predictionES.log << strpr("NNP charge: %16.8E\n", predictionES.structure.charge);
    predictionES.log << strpr("NNP dipole: %16.8E %16.8E %16.8E\n",
                            predictionES.structure.dipole[0],
                            predictionES.structure.dipole[1],
                            predictionES.structure.dipole[2]
				);
    predictionES.log << "\n";
    predictionES.log << "Writing structure with NNP prediction to \"outputES.data\".\n";
    ofstream file("outputES.data");
    predictionES.structure.writeToFile(&file, false);
    file.close();
    predictionES.log << "\n";
    predictionES.log << "Finished.\n";
    prediction.log << "*****************************************""**************************************\n";
    logFile.close();

    return 0;
}
