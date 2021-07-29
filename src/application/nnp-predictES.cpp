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
             << "       - inputES.nn (NNP settings)\n"
             << "       - \"weightsES.%%03d.data\" (weights files for ElectroStatic)\n";
        return 1;
    }


    ofstream logFile;
    logFile.open("nnp-predictES.log");
    PredictionES prediction;
    prediction.log.registerStreamPointer(&logFile);
    prediction.setup();
    prediction.log << "\n";
    prediction.log << "*** PREDICTION **************************"
                      "**************************************\n";
    prediction.log << "\n";
    prediction.log << "Reading structure file...\n";
    prediction.readStructureFromFile("input.data");
    Structure& s = prediction.structure;
    prediction.log << strpr("Structure contains %d atoms (%d elements).\n",
                            s.numAtoms, s.numElements);
    prediction.log << "Calculating NNP prediction...\n";
    prediction.predict();
    prediction.log << "\n";
    prediction.log << "-----------------------------------------"
                      "--------------------------------------\n";
    prediction.log << strpr("NNP charge: %16.8E\n", prediction.structure.charge);
    prediction.log << strpr("NNP charge/atom: %16.8E\n", prediction.structure.charge/s.numAtoms);
    prediction.log << strpr("NNP dipole: %16.8E %16.8E %16.8E\n",
                            prediction.structure.dipole[0],
                            prediction.structure.dipole[1],
                            prediction.structure.dipole[2]
				);
    prediction.log << strpr("NNP dipole/atom: %16.8E %16.8E %16.8E\n",
                            prediction.structure.dipole[0]/s.numAtoms,
                            prediction.structure.dipole[1]/s.numAtoms,
                            prediction.structure.dipole[2]/s.numAtoms
				);
    prediction.log << "\n";
    prediction.log << "Writing structure with NNP prediction to \"outputES.data\".\n";
    ofstream file("outputES.data");
    prediction.structure.writeToFile(&file, false);
    file.close();

    prediction.log << "\n";
    prediction.log << "Finished.\n";
    prediction.log << "*****************************************"
                      "**************************************\n";
    logFile.close();

    return 0;
}
