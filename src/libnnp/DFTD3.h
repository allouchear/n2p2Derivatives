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

#ifndef DFTD3_H
#define DFTD3_H

#include "Structure.h"
#include "Element.h"
#include <vector>
#include <string>

extern"C" {
void  dftd3c_(char* func, int* version, int* nAtoms, int* atnum, double* coords, double* edisp, double* grads);
void  dftd3cenergy_(char* func, int* version, int* nAtoms, int* atnum, double* coords, double* edisp);
void  dftd3cpbc_(char* func, int* version, int* nAtoms, int* atnum, double* coords, double* latVecs, double* edisp, double* grads, double* stress);
void  dftd3cpbcenergy_(char* func, int* version, int* nAtoms, int* atnum, double* coords, double* latVecs, double* edisp);
}

namespace nnp
{
class DFTD3 
{
public:
    DFTD3();
    DFTD3(std::string func);
    /** Calculate DFTD3 energy and it to that of structure
     *
     * @param[in] structure Input structure.
     * @param[in] forces If `true` calculate also forces
     */
    void                     add(Structure& structure, const std::vector<Element>& elements, bool const energy, bool const forces) const;

protected:
    bool                   init; 
    std::string            functional;
};

}

#endif
