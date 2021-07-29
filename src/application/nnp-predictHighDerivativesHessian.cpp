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

#include "Atom.h"
#include "Prediction.h"
#include "Structure.h"
#include "utility.h"
#include "utilityDerivatives.h"
#include "Derivatives.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#ifdef OS_WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>
#include <io.h>
#else /* OS_WIN32 */
#include <pwd.h>
#include <unistd.h>
#include <sys/times.h>
#endif /* OS_WIN32 */

using namespace std;
using namespace nnp;

#ifndef OS_WIN32
#define TIMER_TICK      60
static clock_t it;
static struct tms itt;
void timing(double& cpu,double &sys)
{
        it=times(&itt);
        cpu=(double) itt.tms_utime / (double) TIMER_TICK;
        sys=(double) itt.tms_stime / (double) TIMER_TICK;
}
#endif

vector< vector<double> > getNumericHessian(Prediction& prediction, double step)
{
	Structure& s = prediction.structure;
	int nAtoms = s.numAtoms;
	vector<double> v(nAtoms);
	vector< vector<double> > gp(3,v);
	vector< vector<double> > gm(3,v);

	vector<double> w(3*nAtoms);
	vector <vector<double>> F(3*nAtoms,w);
	double step2=1/step/2;

	for(int i=0;i<nAtoms;i++)
	for(int k=0;k<3;k++)
	{
		int id=3*i+k;
    		prediction.readStructureFromFile("input.data");
		s.atoms[i].r[k] += step;
    		prediction.predict();
		// predict convert forces in physical units
		for(int ia=0;ia<nAtoms;ia++) for(int ka=0;ka<3;ka++) gp[ka][ia] = -s.atoms[ia].f[ka];

    		prediction.readStructureFromFile("input.data");
		s.atoms[i].r[k] -= step;
    		prediction.predict();
		for(int ia=0;ia<nAtoms;ia++) for(int ka=0;ka<3;ka++) gm[ka][ia] = -s.atoms[ia].f[ka];

		for(int j=0;j<=i;j++)
		{
			for(int c = 0;c<3;c++) 
			{
				int jd = 3*j+c;
				// g are in phys units, so use pysunit for step
				F[id][jd]  = F[jd][id] =  (gp[c][j]-gm[c][j])*step2;
			}
		}
	}
	return F;
}
double* getAnalyticHessian(Prediction& prediction)
{
    return prediction.getHessian();
}
void compareAnalytic1Numeric(Prediction& prediction, double* deriv, vector< vector<double> > F)
{
    Structure& s = prediction.structure;
    char xyz[] = {'x','y','z'};
    prediction.log << strpr("%5s %5s %16s %16s %16s\n",
                                "Atom",
                                "Atom",
                                "Analytic1",
                                "Numeric",
                                "Anal-Num");
    double rms = 0;
    double rmax = 0;
    double rmsA = 0;
    double rmaxA = 0;
    double rmsN = 0;
    double rmaxN = 0;
    int n=0;
    for (vector<Atom>::const_iterator it = s.atoms.begin(); it != s.atoms.end(); ++it)
    {
	for(int ic=0;ic<3;ic++)
    		for (vector<Atom>::const_iterator jt = s.atoms.begin(); jt != s.atoms.end(); ++jt)
			for(int jc=0;jc<3;jc++)
			{
				int ii = it->index*3+ic;
				int jj = jt->index*3+jc;
				int idx = (ii<jj)?ii + jj*(jj+1)/2:jj + ii*(ii+1)/2;
    				double dif = deriv[idx]-F[it->index*3+ic][jt->index*3+jc];
        			prediction.log << strpr("%3s%d%c %3s%d%c %16.8E %16.8E %16.8E\n",
                                	prediction.elementMap[it->element].c_str(),it->index + 1, xyz[ic],
                                	prediction.elementMap[jt->element].c_str(),jt->index + 1, xyz[jc],
					deriv[idx], F[it->index*3+ic][jt->index*3+jc],
					deriv[idx]-F[it->index*3+ic][jt->index*3+jc]
                                );
				rms += dif*dif;
				rmsA += deriv[idx]*deriv[idx];
				if(rmaxA<abs(deriv[idx])) rmaxA = abs(deriv[idx]);
				rmsN += F[it->index*3+ic][jt->index*3+jc]*F[it->index*3+ic][jt->index*3+jc];
				if(rmaxN<abs(F[it->index*3+ic][jt->index*3+jc])) rmaxN= abs(F[it->index*3+ic][jt->index*3+jc]);
				if(rmax<abs(dif)) rmax = abs(dif);
				n++;
			}
    }
    rms = sqrt(rms/n);
    rmsA = sqrt(rmsA/n);
    rmsN = sqrt(rmsN/n);
    prediction.log << strpr("\n");
    prediction.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","RMSD="," ",rmsA,rmsN,rms, abs(rmsA-rmsN)/rmsN*100,'%');
    prediction.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","UMAX="," ",rmaxA, rmaxN,rmax, abs(rmaxA-rmaxN)/rmaxN*100,'%');
    prediction.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";

}
void compareAnalytic1Analytic2(Prediction& prediction, double* deriv, Derivatives& deriv2)
{
    Structure& s = prediction.structure;
    char xyz[] = {'x','y','z'};
    prediction.log << strpr("%5s %5s %16s %16s %16s\n",
                                "Atom",
                                "Atom",
                                "Analytic1",
                                "Analytic2",
                                "Anal-Anal");
    double rms = 0;
    double rmax = 0;
    double rmsA = 0;
    double rmaxA = 0;
    double rmsB = 0;
    double rmaxB = 0;
    int n=0;
    for (vector<Atom>::const_iterator it = s.atoms.begin(); it != s.atoms.end(); ++it)
    {
	for(int ic=0;ic<3;ic++)
    		for (vector<Atom>::const_iterator jt = s.atoms.begin(); jt != s.atoms.end(); ++jt)
			for(int jc=0;jc<3;jc++)
			{
				int ii = it->index*3+ic;
				int jj = jt->index*3+jc;
				int idx = (ii<jj)?ii + jj*(jj+1)/2:jj + ii*(ii+1)/2;
    				double d1 = deriv[idx];
    				double d2 = deriv2(ii,jj);
    				double dif = d1-d2;
        			prediction.log << strpr("%3s%d%c %3s%d%c %16.8E %16.8E %16.8E\n",
                                	prediction.elementMap[it->element].c_str(),it->index + 1, xyz[ic],
                                	prediction.elementMap[jt->element].c_str(),jt->index + 1, xyz[jc],
					d1, d2,
					d1-d2
                                );
				rms += dif*dif;
				rmsA += d1*d1;
				if(rmaxA<abs(d1)) rmaxA = abs(d1);
				rmsB += d2*d2;
				if(rmaxB<abs(d2)) rmaxB= abs(d2);
				if(rmax<abs(dif)) rmax = abs(dif);
				n++;
			}
    }
    rms = sqrt(rms/n);
    rmsA = sqrt(rmsA/n);
    rmsB = sqrt(rmsB/n);
    prediction.log << strpr("\n");
    prediction.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","RMSD="," ",rmsA,rmsB,rms, abs(rmsA-rmsB)/rmsB*100,'%');
    prediction.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","UMAX="," ",rmaxA, rmaxB,rmax, abs(rmaxA-rmaxB)/rmaxB*100,'%');
    prediction.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <step>\n"
             << "       <step> ... step for numerical integration\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights.%%03d.data\" (weights files)\n";
        return 1;
    }

    ofstream logFile;
    logFile.open("nnp-predict.log");
    Prediction prediction;
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
                      "---------------------------------------"
                      "---------------------------------------\n";
    prediction.log << strpr("NNP energy: %16.8E\n",
                            prediction.structure.energy);
    prediction.log << "\n";
    prediction.log << "NNP forces:\n";
    for (vector<Atom>::const_iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        prediction.log << strpr("%10zu %2s %16.8E %16.8E %16.8E\n",
                                it->index + 1,
                                prediction.elementMap[it->element].c_str(),
                                it->f[0],
                                it->f[1],
                                it->f[2]);
    }
    prediction.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";
    prediction.log << "\n";
    prediction.log << "NNP 2 derivatives:\n";
    //vector< vector<double> > deriv = prediction.getHessian();
    double cpu, sys;
    double cpun, sysn;
    double cpua, sysa;
    double cpua2, sysa2;
    timing(cpu,sys);
    cerr<<"DEBUG : Begin getAnalyticHessian"<<endl;
    auto deriv = getAnalyticHessian(prediction);
    cerr<<"DEBUG : End getAnalyticHessian"<<endl;
    timing(cpua,sysa);
    cerr<<"DEBUG : Begin prediction.getHighDerivatives"<<endl;
    auto deriv2 = prediction.getHighDerivatives(2,0);
    cerr<<"DEBUG : End prediction.getHighDerivatives"<<endl;
    timing(cpua2,sysa2);
    double step = atof(argv[1]);
    auto F = getNumericHessian(prediction, step);
    timing(cpun,sysn);
	compareAnalytic1Numeric(prediction, deriv,F);
	compareAnalytic1Analytic2(prediction, deriv, deriv2);


    prediction.log << strpr("Time by analytical method             (Hessian)  : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpua-cpu, sysa-sys, cpua-cpu+sysa-sys);
    prediction.log << strpr("Time by analytical method Deriv Class (Hessian)  : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpua2-cpua, sysa2-sysa, cpua2-cpua+sysa2-sysa);
    prediction.log << strpr("Time by numerical method              (Hessian)  : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpun-cpua2, sysn-sysa2, cpun-cpua2+sysn-sysa2);
    prediction.log << "Finished.\n";
    prediction.log << "*****************************************"
                      "***************************************"
                      "***************************************\n";
    logFile.close();

    return 0;
}
