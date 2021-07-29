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
#include "PredictionES.h"
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
void compared2(PredictionES& predictiones, double** deriv1, Derivatives* deriv2, string m1, string m2)
{
    Structure& s = predictiones.structure;
    char xyz[] = {'x','y','z'};
    predictiones.log << strpr("%3s %5s %5s %16s %16s %16s\n",
                                "Mu",
                                "Atom",
                                "Atom",
                                m1.c_str(),
                                m2.c_str(),
                                "Difference");
    double rms = 0;
    double rmax = 0;
    double rmsA = 0;
    double rmaxA = 0;
    double rmsB = 0;
    double rmaxB = 0;
    int n=0;
    for(int im=0;im<3;im++)
    for (vector<Atom>::const_iterator it = s.atoms.begin(); it != s.atoms.end(); ++it)
    {
	for(int ic=0;ic<3;ic++)
    		for (vector<Atom>::const_iterator jt = s.atoms.begin(); jt != s.atoms.end(); ++jt)
			for(int jc=0;jc<3;jc++)
			{
				int ii = it->index*3+ic;
				int jj = jt->index*3+jc;
				int idx = (ii<jj)?ii + jj*(jj+1)/2:jj + ii*(ii+1)/2;
    				double d1 = deriv1[im][idx];
    				double d2 = deriv2[im](ii,jj);
    				double dif = d1-d2;
        			predictiones.log << strpr("%2s%c %3s%d%c %3s%d%c %16.8E %16.8E %16.8E\n",
                                	"Mu", xyz[im],
                                	predictiones.elementMap[it->element].c_str(),it->index + 1, xyz[ic],
                                	predictiones.elementMap[jt->element].c_str(),jt->index + 1, xyz[jc],
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
    predictiones.log << strpr("\n");
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","RMSD="," ",rmsA,rmsB,rms, abs(rmsA-rmsB)/rmsB*100,'%');
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","UMAX="," ",rmaxA, rmaxB,rmax, abs(rmaxA-rmaxB)/rmaxB*100,'%');
    predictiones.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";
}

void compareFirstAnalytic1Analytic2(PredictionES& predictiones, double** deriv1, Derivatives* deriv2)
{
    Structure& s = predictiones.structure;
    char xyz[] = {'x','y','z'};
    predictiones.log << strpr("%5s %5s %16s %16s %16s\n",
                                "Mu",
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
    for(int im=0;im<3;im++)
    for (vector<Atom>::const_iterator it = s.atoms.begin(); it != s.atoms.end(); ++it)
    {
	for(int ic=0;ic<3;ic++)
	{
			int ii = it->index*3+ic;
    			double d1 = deriv1[im][ii];
    			double d2 = deriv2[im](ii);
    			double dif = d1-d2;
        		predictiones.log << strpr("%3s%c %3s%d%c %16.8E %16.8E %16.8E\n",
                               	"Mu", xyz[im],
                               	predictiones.elementMap[it->element].c_str(),it->index + 1, xyz[ic],
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
    predictiones.log << strpr("\n");
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","RMSD="," ",rmsA,rmsB,rms, abs(rmsA-rmsB)/rmsB*100,'%');
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","UMAX="," ",rmaxA, rmaxB,rmax, abs(rmaxA-rmaxB)/rmaxB*100,'%');
    predictiones.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";
}
double** getNumericSecond(PredictionES& predictiones, double step)
{
	Structure& s = predictiones.structure;
	int nAtoms = s.numAtoms;
	int nAtoms3 = 3*nAtoms;
 	int size = nAtoms3*(nAtoms3+1)/2;
	double** d2mu  = new double* [3];
        for(int c=0;c<3;c++) d2mu[c] = new double [size];


	double step2=1/step/2;

	for(int i=0;i<nAtoms;i++)
	for(int k=0;k<3;k++)
	{
		int id=3*i+k;
    		predictiones.readStructureFromFile("input.data");
		s.atoms[i].r[k] += step;
    		auto dmup = predictiones.getdDipole();

    		predictiones.readStructureFromFile("input.data");
		s.atoms[i].r[k] -= step;
    		auto dmum = predictiones.getdDipole();

		for(int j=0;j<=i;j++)
		{
			for(int c = 0;c<3;c++) 
			{
				int jd = 3*j+c;
				int index = id + jd*(jd+1)/2;
				if(id>jd) index = jd + id*(id+1)/2;
				// g are in phys units, so use pysunit for step
				for(int im=0;im<3;im++)
					d2mu[im][index] = (dmup[im][jd]-dmum[im][jd])*step2;
			}
		}
		for(int im=0;im<3;im++)
		{
			if(dmup[im]) delete [] dmup[im];
			if(dmum[im]) delete [] dmum[im];
		}
		if(dmup) delete [] dmup;
		if(dmum) delete [] dmum;
	}
	return d2mu;

}
double**** getNumericThird(PredictionES& predictiones,double step)
{
	int nAtoms = predictiones.structure.numAtoms;
	int nAtoms3 = 3*nAtoms;
	double step2=1.0/step/2.0;

	double**** d3f = new double***[3];
	for(int im=0;im<3;im++)
		d3f[im] = new3Dtable(nAtoms3);

	if(!d3f)
        {
		cerr<<"I cannot create d3f table in getNumericThird"<<endl;
		exit(1);
        }
	/*
	for(int i=0;i<nAtoms3;i++)
	for(int j=0;j<nAtoms3;j++)
	for(int l=0;l<nAtoms3;l++) d3f[i][j][l] =  0;
	*/


	for(int i=0;i<nAtoms;i++)
	for(int k=0;k<3;k++)
	{
		int id=3*i+k;
    		predictiones.readStructureFromFile("input.data");
		predictiones.structure.atoms[i].r[k] += step;
		auto dmup = predictiones.getd2Dipole();

    		predictiones.readStructureFromFile("input.data");
		predictiones.structure.atoms[i].r[k] -= step;
		auto dmum = predictiones.getd2Dipole();

		for(int j=0;j<nAtoms3;j++)
		for(int l=0;l<nAtoms3;l++)
		{
				// g are in phys units, so use pysunit for step
			int jd = j+l*(l+1)/2;
			if(l<j) jd = l+j*(j+1)/2;
			for(int im=0;im<3;im++)
				d3f[im][id][j][l] =  (dmup[im][jd]-dmum[im][jd])*step2;
		}
		for(int im=0;im<3;im++)
                {
			if(dmup[im]) delete [] dmup[im];
			if(dmum[im]) delete [] dmum[im];
                }
                if(dmup) delete [] dmup;
                if(dmum) delete [] dmum;

	}
	return d3f;
}
void compareAnalyticNumericThird(PredictionES& predictiones, double**** d3fNumeric, Derivatives* deriv)
{
    Structure& s = predictiones.structure;
    char xyz[] = {'x','y','z'};
    predictiones.log << strpr("%3s %5s %5s %5s %16s %16s %16s\n",
                                "Mu",
                                "Atom",
                                "Atom",
                                "Atom",
                                "Analytic",
                                "Numeric",
                                "Anal-Num");
    double rms = 0;
    double rmax = 0;
    double rmsA = 0;
    double rmaxA = 0;
    double rmsN = 0;
    double rmaxN = 0;
    int n=0;
    for(int im=0;im<3;im++)
    for (vector<Atom>::const_iterator it = s.atoms.begin(); it != s.atoms.end(); ++it)
    for(int ic=0;ic<3;ic++)
    		for (vector<Atom>::const_iterator jt = s.atoms.begin(); jt != s.atoms.end(); ++jt)
		for(int jc=0;jc<3;jc++)
    			for (vector<Atom>::const_iterator kt = s.atoms.begin(); kt != s.atoms.end(); ++kt)
			for(int kc=0;kc<3;kc++)
			{
				int ii = it->index*3+ic;
				int jj = jt->index*3+jc;
				int kk = kt->index*3+kc;
    				double d1 = deriv[im](ii, jj, kk);
    				double d2 = d3fNumeric[im][ii][jj][kk];
    				double dif = d1-d2;
        			predictiones.log << strpr("%2s%c %3s%d%c %3s%d%c %3s%d%c %16.8E %16.8E %16.8E\n",
                                	"Mu", xyz[im],
                                	predictiones.elementMap[it->element].c_str(),it->index + 1, xyz[ic],
                                	predictiones.elementMap[jt->element].c_str(),jt->index + 1, xyz[jc],
                                	predictiones.elementMap[kt->element].c_str(),kt->index + 1, xyz[kc],
					d1, d2, dif
                                );
				rms += dif*dif;
				rmsA += d1*d1;
				if(rmaxA<abs(d1)) rmaxA = abs(d1);
				rmsN += d2*d2;
				if(rmaxN<abs(d2)) rmaxN= abs(d2);
				if(rmax<abs(dif)) rmax = abs(dif);
				n++;
			}
    rms = sqrt(rms/n);
    rmsA = sqrt(rmsA/n);
    rmsN = sqrt(rmsN/n);
    predictiones.log << strpr("\n");
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","RMSD="," ",rmsA,rmsN,rms, abs(rmsA-rmsN)/rmsN*100,'%');
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","UMAX="," ",rmaxA, rmaxN,rmax, abs(rmaxA-rmaxN)/rmaxN*100,'%');
    predictiones.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";

}
void compareAnalyticNumericFourth(PredictionES& predictiones, double***** d4fNumeric, Derivatives* deriv)
{
    Structure& s = predictiones.structure;
    char xyz[] = {'x','y','z'};
    predictiones.log << strpr("%3s %5s %5s %5s %5s %16s %16s %16s\n",
                                "Mu",
                                "Atom",
                                "Atom",
                                "Atom",
                                "Atom",
                                "Analytic",
                                "Numeric",
                                "Anal-Num");
    double rms = 0;
    double rmax = 0;
    double rmsA = 0;
    double rmaxA = 0;
    double rmsN = 0;
    double rmaxN = 0;
    int n=0;
    for(int im=0;im<3;im++)
    for (vector<Atom>::const_iterator it = s.atoms.begin(); it != s.atoms.end(); ++it)
    for(int ic=0;ic<3;ic++)
    		for (vector<Atom>::const_iterator jt = s.atoms.begin(); jt != s.atoms.end(); ++jt)
		for(int jc=0;jc<3;jc++)
    			for (vector<Atom>::const_iterator kt = s.atoms.begin(); kt != s.atoms.end(); ++kt)
			for(int kc=0;kc<3;kc++)
			{
    				for (vector<Atom>::const_iterator lt = s.atoms.begin(); lt != s.atoms.end(); ++lt)
				for(int lc=0;lc<3;lc++)
				{
				int ii = it->index*3+ic;
				int jj = jt->index*3+jc;
				int kk = kt->index*3+kc;
				int ll = lt->index*3+lc;
    				double d1 = deriv[im](ii, jj, kk,ll);
    				double d2 = d4fNumeric[im][ii][jj][kk][ll];
    				double dif = d1-d2;
        			predictiones.log << strpr("%2s%c %3s%d%c %3s%d%c %3s%d%c %3s%d%c %16.8E %16.8E %16.8E\n",
                                	"Mu", xyz[im],
                                	predictiones.elementMap[it->element].c_str(),it->index + 1, xyz[ic],
                                	predictiones.elementMap[jt->element].c_str(),jt->index + 1, xyz[jc],
                                	predictiones.elementMap[kt->element].c_str(),kt->index + 1, xyz[kc],
                                	predictiones.elementMap[lt->element].c_str(),lt->index + 1, xyz[lc],
					d1, d2, dif
                                );
				rms += dif*dif;
				rmsA += d1*d1;
				if(rmaxA<abs(d1)) rmaxA = abs(d1);
				rmsN += d2*d2;
				if(rmaxN<abs(d2)) rmaxN= abs(d2);
				if(rmax<abs(dif)) rmax = abs(dif);
				n++;
				}
			}
    rms = sqrt(rms/n);
    rmsA = sqrt(rmsA/n);
    rmsN = sqrt(rmsN/n);
    predictiones.log << strpr("\n");
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","RMSD="," ",rmsA,rmsN,rms, abs(rmsA-rmsN)/rmsN*100,'%');
    predictiones.log << strpr("%5s %5s %16.8E %16.8E %16.8E %16.8E %c\n","UMAX="," ",rmaxA, rmaxN,rmax, abs(rmaxA-rmaxN)/rmaxN*100,'%');
    predictiones.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";

}
double***** getNumericFourth(PredictionES& predictiones,double step, int method)
{
	int nAtoms = predictiones.structure.numAtoms;
	int nAtoms3 = 3*nAtoms;
	double step2=1.0/step/2.0;

	double***** d4f = new double**** [3];
	for(int im=0;im<3;im++)
		d4f[im] = new4Dtable(nAtoms3);
	if(!d4f)
        {
		cerr<<"I cannot create d4f table in getNumericFourth"<<endl;
		exit(1);
        }
	/*
	for(int i=0;i<nAtoms3;i++)
	for(int j=0;j<nAtoms3;j++)
	for(int k=0;k<nAtoms3;k++)
	for(int l=0;l<nAtoms3;l++) d4f[i][j][k][l] =  0;
	*/


	for(int i=0;i<nAtoms;i++)
	for(int c=0;c<3;c++)
	{
		int id=3*i+c;
    		predictiones.readStructureFromFile("input.data");
		predictiones.structure.atoms[i].r[c] += step;
		// predict hessian in physical units
		auto gp = predictiones.getHighDerivatives(3,method);

    		predictiones.readStructureFromFile("input.data");
		predictiones.structure.atoms[i].r[c] -= step;
		auto gm = predictiones.getHighDerivatives(3,method);

		for(int j=0;j<nAtoms3;j++)
		for(int k=0;k<nAtoms3;k++)
		for(int l=0;l<nAtoms3;l++)
		{
			for(int im=0;im<3;im++)
				d4f[im][id][j][k][l] =  (gp[im](j,k,l)-gm[im](j,k,l))*step2;
		}
	}
	return d4f;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "USAGE: " << argv[0] << " <step> [method]\n"
             << "       <step> ... step for numerical integration\n"
             << "       [method] optionally 0 (All in memory) , 1(Save in files) or 2(G calculated at each step)\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - inputES.nn (NNP settings)\n"
             << "       - scalingES.data (symmetry function scaling data)\n"
             << "       - \"weightsES.%%03d.data\" (weights files)\n";
        return 1;
    }
    int method = 0;
    if(argc>2) method =  atoi(argv[2]);

    ofstream logFile;
    logFile.open("nnp-predict.log");
    PredictionES predictiones;
    predictiones.log.registerStreamPointer(&logFile);
    predictiones.setup();

    predictiones.log << "\n";
    predictiones.log << "*** PREDICTION **************************"
                      "**************************************\n";
    predictiones.log << "\n";
    predictiones.log << "Reading structure file...\n";
    predictiones.readStructureFromFile("input.data");
    Structure& s = predictiones.structure;
    predictiones.log << strpr("Structure contains %d atoms (%d elements).\n",
                            s.numAtoms, s.numElements);
    predictiones.log << "Calculating NNP predictiones...\n";
    predictiones.predict();
    predictiones.log << "\n";
    predictiones.log << "-----------------------------------------"
                      "--------------------------------------\n";
    predictiones.log << strpr("NNP charge: %16.8E\n", predictiones.structure.charge);
    predictiones.log << strpr("NNP charge/atom: %16.8E\n", predictiones.structure.charge/s.numAtoms);
    predictiones.log << strpr("NNP dipole: %16.8E %16.8E %16.8E\n",
                            predictiones.structure.dipole[0],
                            predictiones.structure.dipole[1],
                            predictiones.structure.dipole[2]
				);
    predictiones.log << strpr("NNP dipole/atom: %16.8E %16.8E %16.8E\n",
                            predictiones.structure.dipole[0]/s.numAtoms,
                            predictiones.structure.dipole[1]/s.numAtoms,
                            predictiones.structure.dipole[2]/s.numAtoms
				);
    predictiones.log << "\n";
    predictiones.log << "-----------------------------------------"
                      "---------------------------------------"
                      "---------------------------------------\n";
    predictiones.log << "\n";
    predictiones.log << "NNP First dipole derivatives:\n";
    double tmpc, tmps;
    double cpuda1, sysda1;
    double cpuda2, sysda2;

    timing(tmpc,tmps);
    //double** Mode::computedDipole(Structure& structure) const
    auto deriv1 = predictiones.getdDipole();
    timing(cpuda1,sysda1); cpuda1 -= tmpc; sysda1 -= tmps;

    timing(tmpc,tmps);
    auto deriv2 = predictiones.getHighDerivatives(1,method);
    timing(cpuda2,sysda2); cpuda2 -= tmpc; sysda2 -= tmps;
    double step = atof(argv[1]);

    compareFirstAnalytic1Analytic2(predictiones, deriv1, deriv2);

    predictiones.log << strpr("Time by analytical method             (First)  : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpuda1, sysda1, cpuda1+sysda1);
    predictiones.log << strpr("Time by analytical method Deriv Class (First)  : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpuda2, sysda2, cpuda2+sysda2);
    predictiones.log << "Finished.\n";
    predictiones.log << "*****************************************"
                      "***************************************"
                      "***************************************\n";
/////////////////////////////////// Second derivatives
    predictiones.log << " Begin Second Numeric method\n";

    timing(tmpc,tmps);
    double cpud2n, sysd2n;
    auto d2muNum = getNumericSecond(predictiones, step);
    timing(cpud2n,sysd2n); cpud2n -= tmpc; sysd2n -= tmps;

    predictiones.log << " Begin d2Dipole First method\n";
    double cpud2a1, sysd2a1;
    timing(tmpc,tmps);
    auto d2mu1 = predictiones.getd2Dipole();
    timing(cpud2a1,sysd2a1); cpud2a1 -= tmpc; sysd2a1 -= tmps;


    predictiones.log << " Begin analytical second derivatives\n";
    double cpud2a2, sysd2a2;
    timing(tmpc,tmps);
    auto d2mu2 = predictiones.getHighDerivatives(2,method);
    timing(cpud2a2,sysd2a2); cpud2a2 -= tmpc; sysd2a2 -= tmps;
    predictiones.log << " Begin compare ana1-anal2\n";
    compared2(predictiones, d2muNum, d2mu2, "Numeric","Analytical/D");
    compared2(predictiones, d2mu1, d2mu2, "Analytical","Analytical/D");
//
/////////////////////////////////// Third derivatives
    predictiones.log << " Begin analytical third derivatives\n";
    double cpud3a2, sysd3a2;
    timing(tmpc,tmps);
    auto d3mu2 = predictiones.getHighDerivatives(3,method);
    timing(cpud3a2,sysd3a2); cpud3a2 -= tmpc; sysd3a2 -= tmps;

    predictiones.log << " Begin third Numeric method\n";

    timing(tmpc,tmps);
    double cpud3n, sysd3n;
    auto d3muNum = getNumericThird(predictiones, step);
    timing(cpud3n,sysd3n); cpud3n -= tmpc; sysd3n -= tmps;

    predictiones.log << " Begin compare num3-anal3\n";
    compareAnalyticNumericThird(predictiones, d3muNum, d3mu2);
//
/////////////////////////////////// Fourth derivatives
    predictiones.log << " Begin analytical Fourth derivatives\n";
    double cpud4a2, sysd4a2;
    timing(tmpc,tmps);
    auto d4mu2 = predictiones.getHighDerivatives(4,method);
    timing(cpud4a2,sysd4a2); cpud4a2 -= tmpc; sysd4a2 -= tmps;

    predictiones.log << " Begin Fourth Numeric method\n";

    timing(tmpc,tmps);
    double cpud4n, sysd4n;
    auto d4muNum = getNumericFourth(predictiones, step, method);
    timing(cpud4n,sysd4n); cpud4n -= tmpc; sysd4n -= tmps;

    predictiones.log << " Begin compare num4-anal4\n";
    compareAnalyticNumericFourth(predictiones, d4muNum, d4mu2);



    predictiones.log << strpr("Time by analytical method             (First)    : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpuda1, sysda1, cpuda1+sysda1);
    predictiones.log << strpr("Time by analytical method Deriv Class (First)    : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpuda2, sysda2, cpuda2+sysda2);
    predictiones.log << strpr("Time by analytical method             (Second)   : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud2a1, sysd2a1, cpud2a1+sysd2a1);
    predictiones.log << strpr("Time by analytical method Deriv Class (Second)   : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud2a2, sysd2a2, cpud2a2+sysd2a2);
    predictiones.log << strpr("Time by numerical  method             (Second)   : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud2n, sysd2n, cpud2n+sysd2n);
    predictiones.log << strpr("Time by analytical method Deriv Class (Second)   : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud2a2, sysd2a2, cpud2a2+sysd2a2);
    predictiones.log << strpr("Time by numerical  method             (Third)    : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud3n, sysd3n, cpud3n+sysd3n);
    predictiones.log << strpr("Time by analytical method Deriv Class (Third)    : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud3a2, sysd3a2, cpud3a2+sysd3a2);
    predictiones.log << strpr("Time by numerical  method             (Fourth)   : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud4n, sysd4n, cpud4n+sysd4n);
    predictiones.log << strpr("Time by analytical method Deriv Class (Fourth)   : cpu = %16.8E sys = %16.8E all = %16.8E\n",cpud4a2, sysd4a2, cpud4a2+sysd4a2);
    predictiones.log << "Finished.\n";
    predictiones.log << "*****************************************"
                      "***************************************"
                      "***************************************\n";

    logFile.close();

    return 0;
}
