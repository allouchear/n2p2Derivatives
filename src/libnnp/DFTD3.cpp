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

#include "DFTD3.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>

using namespace std;
using namespace nnp;

DFTD3::DFTD3() : init                 (false),
               functional            ("None")
{
}
DFTD3::DFTD3(std::string func ) : init                 (true),
                 functional           (func)
{
	if(func=="None") init = false;
}

void DFTD3::add(Structure& structure,  const std::vector<Element>& elements, bool const energy, bool const forces) const
{
   double edisp = 0.0;
   int nAtoms;
   int* atnum = NULL;
   double* grads = NULL;
   double stress[9];
   double latVecs[9];
   double* coords;
   int version = 4;
   Atom* a;

   if(!init) return;

   nAtoms = structure.atoms.size();
   atnum = new int [nAtoms];
   for(int ia=0;ia<nAtoms;ia++)
   {
        a = &(structure.atoms.at(ia));
	atnum[ia] = (int)elements.at(a->element).getAtomicNumber();
   }
   coords = new double [3*nAtoms];
   int k = 0;
   for(int i=0;i<3;i++)
   for(int ia=0;ia<nAtoms;ia++)
   {
      a = &(structure.atoms.at(ia));
      coords [k] = a->r[i]; 
      k++;
    }
   if(!structure.isPeriodic)
   {
      if(forces)
      {
   	  grads = new double [3*nAtoms];
         dftd3c_((char*)functional.c_str(),&version, &nAtoms, atnum, coords, &edisp, grads);
      }
      else dftd3cenergy_((char*) functional.c_str(),&version, &nAtoms, atnum, coords, &edisp);
   }
   else
   {
      int k = 0;
      for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
      {
          latVecs [k] = structure.box[i][j]; 
          k++;
      }
      if(forces)
      {
   	  grads = new double [3*nAtoms];
         dftd3cpbc_((char*) functional.c_str(),&version, &nAtoms, atnum, coords, latVecs, &edisp, grads, stress);
      }
      else dftd3cpbcenergy_((char*) functional.c_str(),&version, &nAtoms, atnum, coords, latVecs, &edisp);
   }
   if(energy) structure.energy += edisp;
   if(forces)
   {
     k = 0;
     for(int i=0;i<3;i++)
     for(int ia=0;ia<nAtoms;ia++)
     {
       a = &(structure.atoms.at(ia));
       a->f[i] -= grads[k]; 
       k++;
     }
   }
   if(coords) delete [] coords;
   if(grads) delete [] grads;
   if(atnum) delete [] atnum;
}
