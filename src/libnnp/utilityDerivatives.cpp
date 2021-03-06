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

#include "utility.h"
#include "utilityDerivatives.h"
#include <algorithm> // std::max
#include <cstdio>    // vsprintf
#include <cstdarg>   // va_list, va_start, va_end
#include <iomanip>   // std::setw
#include <sstream>   // std::istringstream
#include <stdexcept> // std::runtime_error
#include <cmath>
#include <iostream>
#include "CutoffFunction.h"

#ifdef HIGH_DERIVATIVES

#define STRPR_MAXBUF 1024

using namespace std;

namespace nnp
{

double* new1Dtable(int size)
{
    double* t = new double[size];
    return t;
}
void free1Dtable(double* t)
{
    delete[] t;
}
double** new2Dtable(int size)
{
    double** t = new double*[size];
    for (int l = 0; l < size; l++) t[l]  = new double[size];
    return t;
}
void free2Dtable(int size, double**t)
{
    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
double*** new3Dtable(int size)
{
    double*** t = new double**[size];
    for (int l = 0; l < size; l++) 
		t[l] = new double*[size];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m < size; m++) 
		t[l][m] = new double[size];
    return t;
}
void free3Dtable(int size, double***t)
{
    for (int l = 0; l < size; l++) 
    	for (int m = 0; m < size; m++) 
		delete [] t[l][m];
    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
double**** new4Dtable(int size)
{
    double**** t = new double***[size];
    for (int l = 0; l < size; l++) 
		t[l] = new double**[size];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m < size; m++) 
		t[l][m] = new double*[size];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m < size; m++) 
    		for (int n = 0; n < size; n++) 
		t[l][m][n] = new double[size];
    return t;
}
void free4Dtable(int size, double****t)
{
    for (int l = 0; l < size; l++) 
    	for (int m = 0; m < size; m++) 
    		for (int n = 0; n < size; n++) 
		delete [] t[l][m][n];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m < size; m++) 
		delete [] t[l][m];

    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
static void swap(int *a, int *b)
{
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
//permutation function
void permutation(int *arr, int start, int end, double val, double*** tab3)
{
    if(start==end)
    {
	tab3[arr[0]][arr[1]][arr[2]] = val;
        return;
    }
    for(int i=start;i<=end;i++)
    {
        //swapping numbers
        swap((arr+i), (arr+start));
        //fixing one first digit
        //and calling permutation on
        //the rest of the digits
        permutation(arr, start+1, end, val, tab3);
        swap((arr+i), (arr+start));
    }
}
//permutation function
void permutation(int *arr, int start, int end, double val, double**** tab4)
{
    if(start==end)
    {
	tab4[arr[0]][arr[1]][arr[2]][arr[3]] = val;
        return;
    }
    for(int i=start;i<=end;i++)
    {
        //swapping numbers
        swap((arr+i), (arr+start));
        //fixing one first digit
        //and calling permutation on
        //the rest of the digits
        permutation(arr, start+1, end, val, tab4);
        swap((arr+i), (arr+start));
    }
}

double getDerivativesR(const double* dr, const double& r, int ialpha, int order)
{
	if(order==0) return r;
	else if(order==1)
	{
		return dr[ialpha]/r;
		
	}
	else if(order==2)
	{
		double rr=1.0/r;
		double rr3=rr*rr*rr;
		return -dr[ialpha]*dr[ialpha]*rr3+rr;
	}
	else if(order==3)
	{
		double rr=1.0/r;
		double rr2=rr*rr;
		double rr3=rr2*rr;
		double rr5=rr3*rr2;
		double x3 = dr[ialpha]*dr[ialpha]*dr[ialpha];
		return 3*x3*rr5-3*dr[ialpha]*rr3;
	}
	else if(order==4)
	{
		double rr=1.0/r;
		double rr2=rr*rr;
		double rr3=rr2*rr;
		double rr5=rr3*rr2;
		double rr7=rr5*rr2;
		double x2 = dr[ialpha]*dr[ialpha];
		double x4 = x2*x2;
		return -15*x4*rr7+18*x2*rr5-3*rr3;
	}
	else if(order==5)
	{
		double rr=1.0/r;
		double rr2=rr*rr;
		double rr3=rr2*rr;
		double rr5=rr3*rr2;
		double rr7=rr5*rr2;
		double rr9=rr7*rr2;
		double x = dr[ialpha];
		double x3 = x*x*x;
		double x5 = x3*x*x;
		return 105*x5*rr9-150*x3*rr7+45*x*rr5;
	}
	else if(order==6)
	{
		double rr=1.0/r;
		double rr2=rr*rr;
		double rr3=rr2*rr;
		double rr5=rr3*rr2;
		double rr7=rr5*rr2;
		double rr9=rr7*rr2;
		double rr11=rr9*rr2;
		double x = dr[ialpha];
		double x2 = x*x;
		double x4 = x2*x2;
		double x6 = x4*x2;
		return -945*x6*rr11+1575*x4*rr9-675*x2*rr7+45*rr5;
	}
	else
	{
		throw invalid_argument("ERROR: Derivatives over 6 are not yet implemented for r function.\n");
	}
}
double getDerivativesR(const double* dr, const double& r, int ialpha, int orderi, int jalpha, int orderj)
{
	if(ialpha==jalpha) return getDerivativesR(dr,r,ialpha, orderi+orderj);
	else if(orderi==0) return getDerivativesR(dr,r,jalpha, orderj);
	else if(orderj==0) return getDerivativesR(dr,r,ialpha, orderi);
	if(orderi>orderj)
	{
		int t = orderi;
		orderi = orderj;
		orderj = t;
		t= ialpha;
		ialpha = jalpha;
		jalpha = t;
	}
	if(orderi+orderi<=6)
	{
		if(orderi==1) 
		{
			if(orderj==1 ) 
			{
				double rr=1.0/r;
				double rr3=rr*rr*rr;
				double xixj = dr[ialpha]*dr[jalpha];
				return -xixj*rr3;
			}
			else if(orderj==2)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double xi = dr[ialpha];
				double xixj2 = dr[ialpha]*dr[jalpha]*dr[jalpha];
				return 3*xixj2*rr5-xi*rr3;
			}
			else if(orderj==3)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double xixj = dr[ialpha]*dr[jalpha];
				double xixj3 = xixj*dr[jalpha]*dr[jalpha];
				return -15*xixj3*rr7+9*xixj*rr5;
			}
			else if(orderj==4)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double rr9=rr7*rr2;
				double xi = dr[ialpha];
				double xixj = dr[ialpha]*dr[jalpha];
				double xixj2 = xixj*dr[jalpha];
				double xixj4 = xixj2*dr[jalpha]*dr[jalpha];
				return 105*xixj4*rr9-45*xixj2*rr7+9*xi*rr5;
			}
			else if(orderj==5)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double rr9=rr7*rr2;
				double rr11=rr9*rr2;
				double xixj = dr[ialpha]*dr[jalpha];
				double xixj3 = xixj *dr[jalpha]*dr[jalpha];
				double xixj5 = xixj3*dr[jalpha]*dr[jalpha];
				return -945*xixj5*rr11+525*xixj3*rr9-225*xixj*rr7;
			}
		}
		else if(orderi==2) 
		{
			if(orderj==2)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double xi2 = dr[ialpha]*dr[ialpha];
				double xj2 = dr[jalpha]*dr[jalpha];
				return -15*xi2*xj2*rr7+3*(xi2+xj2)*rr5+rr3;
			}
			else if(orderj==3)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double rr9=rr7*rr2;
				double xi2 = dr[ialpha]*dr[ialpha];
				double xj = dr[jalpha];
				double xj3 = dr[jalpha]*dr[jalpha]*dr[jalpha];
				return 105*xi2*xj3*rr9-45*xi2*xj*rr7-15*xj3*rr7+9*xj*rr5;
			}
			else if(orderj==4)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double rr9=rr7*rr2;
				double rr11=rr9*rr2;
				double xi2 = dr[ialpha]*dr[ialpha];
				double xj2 = dr[jalpha]*dr[jalpha];
				double xj4 = xj2*xj2;
				double xi2xj2 = xi2*xj2;
				double xi2xj4 = xi2xj2*xj2;
				return -945*xi2xj4*rr11+630*xi2xj2*rr9-45*xi2*rr7+105*xj4*rr9-45*xj2*rr7+9*rr5;
			}
		}
		else if(orderi==3) 
		{
			if(orderj==3)
			{
				double rr=1.0/r;
				double rr2=rr*rr;
				double rr3=rr2*rr;
				double rr5=rr3*rr2;
				double rr7=rr5*rr2;
				double rr9=rr7*rr2;
				double rr11=rr9*rr2;
				double xi = dr[ialpha];
				double xj = dr[jalpha];
				double xi3 = xi*xi*xi;
				double xj3 = xj*xj*xj;
				return -945*xi3*xj3*rr11+630*(xi3*xj+xi*xj3)*rr9-135*xi*xj*rr7;
			}
		}
		
	}
	else
	{
		throw invalid_argument("ERROR: Derivatives over 6 are not yet implemented for r function.\n");
	}
	return 0;
}
double getDerivativesRadial(double x, double eta, int order)
{
	if(order==0) return exp(-eta * x*x);
	else if(order==1)
	{
		return -2 * eta * x* exp(-eta *x*x);
	}
	else if(order==2)
	{
		double t1 = x * x;
		double t3 = exp(-eta * t1);
		double t4 = eta * eta;
		return  4 * t4 * t1 * t3 - 2 * eta * t3;
	}
	else if(order==3)
	{
		double t1 = eta * eta;
		double t3 = x * x;
		double t5 = exp(-eta * t3);
		return -0.8e1 * t1 * eta * t3 * x * t5 + 0.12e2 * t1 * x * t5;
	}
	else if(order==4)
	{
		double t1 = x * x;
		double t3 = exp(-eta * t1);
		double t4 = eta * eta;
		double t11 = t4 * t4;
		double t12 = t1 * t1;
		return -0.48e2 * t4 * eta * t1 * t3 + 0.16e2 * t11 * t12 * t3 + 0.12e2 * t3 * t4;
	}
	else if(order==5)
	{
		double t1 = eta * eta;
		double t4 = x * x;
		double t6 = exp(-eta * t4);
		double t9 = t1 * t1;
		double t15 = t4 * t4;
		return -0.32e2 * t9 * eta * t15 * x * t6 - 0.120e3 * t1 * eta * x * t6 + 0.160e3 * t9 * t4 * x * t6;

	}
	else if(order==6)
	{
		double t1 = x * x;
		double t3 = exp(-eta * t1);
		double t4 = eta * eta;
		double t8 = t4 * t4;
		double t13 = t1 * t1;
		return 0.64e2 * t8 * t4 * t13 * t1 * t3 - 0.480e3 * t8 * eta * t13 * t3 - 0.120e3 * t3 * t4 * eta + 0.720e3 * t8 * t1 * t3;
	}
	else
	{
		throw invalid_argument("ERROR: Derivatives over 6 are not yet implemented for exp(-eta*x*x).\n");
	}
}
double getDerivativesG(const double* rijv, double eta, double rs, double secondDeriv[][6], const CutoffFunction& cf)
{
        // Energy calculation.
	double rij = 0;
	for(size_t i=0;i<3;i++) rij += rijv[i]*rijv[i];
	rij = sqrt(rij);
	double x = (rij - rs);
	double d2fc = cf.dnf(rij, 2);
	double d1fc = cf.dnf(rij, 1);
	double d0fc = cf.f(rij);

	double d2g = getDerivativesRadial(x, eta, 2);
	double d1g = getDerivativesRadial(x, eta, 1);
	double d0g = exp(-eta*x*x);

	double d1F =  d1fc*d0g + d0fc*d1g;
	double d2F =  d2fc*d0g + d0fc*d2g + 2*d1fc*d1g;

	double d1r[3];
	for(int ia=0;ia<3;ia++) d1r[ia] = getDerivativesR(rijv, rij, ia, 1);

	for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			double diaja = getDerivativesR(rijv, rij, ia, 1, ja,1);
			double ddiaja = d1r[ia]*d1r[ja];
			double d2G = (d1F*diaja + d2F*ddiaja);
			secondDeriv[ia][ja] = d2G;
			secondDeriv[ia+3][ja+3] = d2G;
			secondDeriv[ia][ja+3] = -d2G;
			secondDeriv[ia+3][ja] = -d2G;
		}
	return d0fc*d0g;
}
/* firstDeriv vector with 9 elements */
/* Derivatives of Rij dot Rik*/
double getDerivativesRijk(const double* rij, const double* rik, double firstDeriv[])
{
	// rij[l] = ri[l]-rj[l]
	for(size_t l=0;l<3;l++) firstDeriv[l] = rij[l]+rik[l];// d/dxi, d/dyi, d/dzi
	for(size_t l=3;l<6;l++) firstDeriv[l] = -rik[l-3];// d/dxj, d/dyj, d/dzj
	for(size_t l=6;l<9;l++) firstDeriv[l] = -rij[l-6];// d/dxk, d/dyk, d/dzk
	double rijsrik = 0;
	for(size_t l=0;l<3;l++) rijsrik += rij[l]*rik[l];// rij dot rik
	return rijsrik;
}
/* secondDeriv matrix with 9x9 elements */
void getDerivativesRijk(double secondDeriv[][9])
{
	for(size_t l=0;l<9;l++)  
		for(size_t m=0;m<9;m++)
			secondDeriv[l][m] = 0;

	for(size_t l=0;l<3;l++)  secondDeriv[l][l] = 2; // d/dxi2, d/dyi2, d/dzi2
	/*
	firstDeriv[0][0+3] = -1; // d/dxidxj
	firstDeriv[0][0+6] = -1; // d/dxidxk
	firstDeriv[1][1+3] = -1; // d/dyidyj
	firstDeriv[1][1+6] = -1; // d/dyidyk
	firstDeriv[2][2+3] = -1; // d/dzidzj
	firstDeriv[2][2+6] = -1; // d/dzidzk

	firstDeriv[3][3+3] = 1; // d/dxjdxk
	firstDeriv[4][4+3] = 1; // d/dyjdyk
	firstDeriv[5][5+3] = 1; // d/dzjdzk
	*/
	for(size_t l=0;l<3;l++)  secondDeriv[l][l+3] = secondDeriv[l+3][l] = secondDeriv[l][l+6] = secondDeriv[l+6][l] = -1;
	for(size_t l=0;l<3;l++)  secondDeriv[l+3][l+6] = secondDeriv[l+6][l+3] = 1;
}
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
double getDerivativesRijk(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9])
{
	double rijsrik = getDerivativesRijk(rij, rik, firstDeriv);
	getDerivativesRijk(secondDeriv);
	return rijsrik;

}
/* firstDeriv vector with 6 elements */
double getDerivativesRij(const double* rij,  double firstDeriv[])
{
	double r = 0;
	for(size_t l=0;l<3;l++)   r += rij[l]*rij[l];
	r = sqrt(r);
	for(size_t l=0;l<3;l++)   firstDeriv[l] = getDerivativesR(rij, r, l,1);// d/dxi, d/dyi, d/dzi
	for(size_t l=3;l<6;l++)   firstDeriv[l] = -firstDeriv[l-3];// d/dxj, d/dyj, d/dzj
	return r;
}
/* secondDeriv matrix with 6x6 elements */
double getDerivativesRij(const double* rij,  double secondDeriv[][6])
{
	double firstDeriv[6];
	double r = getDerivativesRij(rij,  firstDeriv); 
	double sr = 1/r;
	double sr2 = sr*sr;
	for(size_t l=0;l<3;l++)   
	for(size_t m=l;m<3;m++)   secondDeriv[l][m] = secondDeriv[m][l] = getDerivativesR(rij, r, l,1,m,1);

 	double rji[3];
	for(size_t l=0;l<3;l++)   rji[l] = - rij[l];
	for(size_t l=0;l<3;l++)   
	for(size_t m=l;m<3;m++)   secondDeriv[l+3][m+3] = secondDeriv[m+3][l+3] = getDerivativesR(rji, r, l,1,m,1);


	for(size_t l=0;l<3;l++)   
	for(size_t m=0+3;m<3+3;m++)   
	{
		secondDeriv[l][m] = -rij[l]*sr2*firstDeriv[m];
		if(m-3==l) secondDeriv[l][m] -= sr;
		secondDeriv[m][l] = secondDeriv[l][m];
	}

	return r;
}
/* firstDeriv vector with 6 elements */
/* secondDeriv matrix with 6x6 elements */
double getDerivativesRij(const double* rij,  double firstDeriv[], double secondDeriv[][6])
{
	double r = getDerivativesRij(rij,  firstDeriv); 
	double sr = 1/r;
	double sr2 = sr*sr;
	for(size_t l=0;l<3;l++)   
	for(size_t m=l;m<3;m++)   secondDeriv[l][m] = secondDeriv[m][l] = getDerivativesR(rij, r, l,1,m,1);

	for(size_t l=0;l<3;l++)   
	for(size_t m=l;m<3;m++)   secondDeriv[l+3][m+3] = secondDeriv[m+3][l+3] = secondDeriv[l][m];

	for(size_t l=0;l<3;l++)   
	for(size_t m=0+3;m<3+3;m++)   
	{
		secondDeriv[l][m] = -rij[l]*sr2*firstDeriv[m];
		if(m-3==l) secondDeriv[l][m] -= sr;
	}
	for(size_t l=3;l<3+3;l++)   
	for(size_t m=0;m<3;m++)   
		secondDeriv[l][m] = secondDeriv[l-3][m+3];

	return r;
}
static void build9From6ij(const double d6[], double d9[])
{
	for(size_t l=0;l<6;l++) d9[l] = d6[l];
	for(size_t l=6;l<9;l++) d9[l] = 0;
}
static void build9From6ik(const double d6[], double d9[])
{
	for(size_t l=0;l<3;l++) d9[l] = d6[l];
	for(size_t l=3;l<6;l++) d9[l] = 0;
	for(size_t l=6;l<9;l++) d9[l] = d6[l-3];
}
static void build9From6jk(const double d6[], double d9[])
{
	for(size_t l=0;l<3;l++) d9[l] = 0;
	for(size_t l=3;l<9;l++) d9[l] = d6[l-3];
}
static void build9From6ij(const double d6[][6], double d9[][9])
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) d9[l][m] = 0;
	for(size_t l=0;l<6;l++) for(size_t m=0;m<6;m++) d9[l][m] = d6[l][m];
}
static void build9From6ik(const double d6[][6], double d9[][9])
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) d9[l][m] = 0;
	for(size_t l=0;l<3;l++) for(size_t m=0;m<3;m++) d9[l][m] = d6[l][m];
	for(size_t l=0;l<3;l++) for(size_t m=6;m<9;m++) d9[l][m] = d9[m][l] = d6[l][m-3];
	for(size_t l=6;l<9;l++) for(size_t m=6;m<9;m++) d9[l][m] = d6[l-3][m-3];
}
static void build9From6jk(const double d6[][6], double d9[][9])
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) d9[l][m] = 0;
	for(size_t l=3;l<9;l++) for(size_t m=3;m<9;m++) d9[l][m] = d6[l-3][m-3];
}
/* firstDeriv vector with 9 elements : derivatives of |rij||rik| */
// variables : xi, yi, zi, xj, yj, zj, xk, yk, zk
double getDerivativesRijRik(const double* rij, const double* rik, double firstDeriv[])
{
	double drij[9];
	double rijm = getDerivativesRij(rij,  drij);
	for(size_t l=6;l<9;l++) drij[l] = 0;

	double drik[9];
	double rikm = getDerivativesRij(rik,  drik);
	for(size_t l=3;l<6;l++) drik[l+3] = drik[l]; // shift k : 3=>6
	for(size_t l=3;l<6;l++) drik[l] = 0.0; // drik/dxj=0

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = drij[l]*rikm+rijm*drik[l];

	return rijm*rikm;
}
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
double getDerivativesRijRik(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9])
{
	double dr6[6];
	double d2r6[6][6];

	double drij[9];
	double d2rij[9][9];
	double rijm = getDerivativesRij(rij,  dr6, d2r6);
	build9From6ij(dr6, drij);
	build9From6ij(d2r6, d2rij);

	double drik[9];
	double d2rik[9][9];
	double rikm = getDerivativesRij(rik,  dr6, d2r6);

	build9From6ik(dr6, drik);
	build9From6ik(d2r6, d2rik);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = drij[l]*rikm+drik[l]*rijm;

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] =  
				 d2rij[l][m] * rikm + drij[l]*drik[m] +drij[m]*drik[l] + rijm*d2rik[l][m];

	return rijm*rikm;
}
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
double getDerivativesRijRikm1(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9])
{
	// g = RijRik ; f = 1/g = RijRik^-1
	double dg[9];
	double d2g[9][9];
	double g  = getDerivativesRijRik(rij, rik, dg, d2g);
	double gm1 = 1/g;
	double gm2 = gm1*gm1;
	double gm3 = gm2*gm1;

	for(size_t l=0;l<9;l++) firstDeriv[l] = -gm2*dg[l]; 
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) 
		secondDeriv[l][m] = 2*gm3*dg[l]*dg[m]-gm2*d2g[l][m];

	return gm1;
}
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
// derivatives of cos theatijk
double getDerivativescosijk(const double* rij, const double* rik,double firstDeriv[],double secondDeriv[][9])
{
	// f = Rij dot Rik ; g = RijRik^-1
	double df[9];
	double d2f[9][9];
	double f =  getDerivativesRijk(rij, rik, df,d2f);


	double dg[9];
	double d2g[9][9];
	double g =  getDerivativesRijRikm1(rij, rik, dg,d2g);

	for(size_t l=0;l<9;l++) firstDeriv[l] = df[l]*g + f*dg[l];
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) 
		secondDeriv[l][m] = d2f[l][m]*g + df[l]*dg[m] + df[m]*dg[l]+ f*d2g[l][m];

	return f*g;
}
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
// derivatives of (1+ lambda * cos theatijk)^xi
double getDerivativescosijk(const double* rij, const double* rik, double lambda, double zeta, double firstDeriv[],double secondDeriv[][9])
{
	double dc[9];
	double d2c[9][9];
	double c = getDerivativescosijk(rij, rik, dc, d2c);

	double plambda = (1+lambda*c);
	double ccx = 0;
	if(plambda>0)
	{
		ccx = pow(plambda,zeta);
		double ccxm1 = ccx/plambda;
		double ccxm2 = ccxm1/plambda;
		double lx = lambda*zeta;
		for(size_t l=0;l<9;l++) firstDeriv[l] = lx*ccxm1*dc[l];
		for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) 
			secondDeriv[l][m] = lx*(zeta-1)*lambda*ccxm2*dc[l]*dc[m] + lx*ccxm1*d2c[l][m];
	}
	else
	{
		for(size_t l=0;l<9;l++) firstDeriv[l] = 0;
		for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) 
			secondDeriv[l][m] = 0;
	}
	return ccx;
}	
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
/* derivatives of Rij^2 + Rik^2 */
double getDerivativesRij2pRik2(const double* rij, const double* rik, double rs, double firstDeriv[],double secondDeriv[][9])
{
	double df6[6];
	double d2f6[6][6];

	double drij[9];
	double d2rij[9][9];
	double rijm = getDerivativesRij(rij,  df6, d2f6)-rs;
	build9From6ij(df6, drij);
	build9From6ij(d2f6, d2rij);

	double drik[9];
	double d2rik[9][9];
	double rikm = getDerivativesRij(rik,  df6, d2f6)-rs;
	build9From6ik(df6, drik);
	build9From6ik(d2f6, d2rik);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = 2*(drij[l]*rijm+drik[l]*rikm);

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] =  
				2*(drij[l]*drij[m]+drik[l]*drik[m])
				+2*(rijm*d2rij[l][m]+rikm*d2rik[l][m]);

	return rijm*rijm+rikm*rikm;
}
double getDerivativesexpetaRij2pRik2(const double* rij, const double* rik,double eta, double rs, double firstDeriv[],double secondDeriv[][9])
{
	// f = Rij^2 + Rik^2
	// g = exp(-eta*f), derivative of g
	double df[9];
	double d2f[9][9];
	double f = getDerivativesRij2pRik2(rij, rik, rs, df, d2f);

	double g = exp(-eta*f);
	double eg = -eta*exp(-eta*f);

	for(size_t l=0;l<9;l++) firstDeriv[l] = eg*df[l];
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) 
		secondDeriv[l][m] = -eta*eg* df[l]*df[m] + eg*d2f[l][m];

	return g;
}
/* secondDeriv matrix with 9x9 elements */
/* firstDeriv vector with 9 elements */
/* derivatives of (Rij-rs)^2 + (Rik-rs)^2 + (Rjk-rs)^2 */
double getDerivativesRij2pRik2pRjk2(const double* rij, const double* rik, const double* rjk, double rs, double firstDeriv[],double secondDeriv[][9])
{
	double df6[6];
	double d2f6[6][6];

	double drij[9];
	double d2rij[9][9];
	double rijm = getDerivativesRij(rij,  df6, d2f6)-rs;
	build9From6ij(df6, drij);
	build9From6ij(d2f6, d2rij);

	double drik[9];
	double d2rik[9][9];
	double rikm = getDerivativesRij(rik,  df6, d2f6)-rs;
	build9From6ik(df6, drik);
	build9From6ik(d2f6, d2rik);

	double drjk[9];
	double d2rjk[9][9];
	double rjkm = getDerivativesRij(rjk,  df6, d2f6)-rs;
	build9From6jk(df6, drjk);
	build9From6jk(d2f6, d2rjk);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = 2*(drij[l]*rijm+drik[l]*rikm +drjk[l]*rjkm);

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] =  
				2*(drij[l]*drij[m]+drik[l]*drik[m]+drjk[l]*drjk[m])
				+2*(rijm*d2rij[l][m]+rikm*d2rik[l][m]+rjkm*d2rjk[l][m]);

	return rijm*rijm+rikm*rikm+rjkm*rjkm;
}
double getDerivativesexpetaRij2pRik2pRjk2(const double* rij, const double* rik, const double* rjk, double eta, double rs, double firstDeriv[],double secondDeriv[][9])
{
	// f = Rij^2 + Rik^2 + Rjk^2
	// g = exp(-eta*f), derivative of g
	double df[9];
	double d2f[9][9];
	double f = getDerivativesRij2pRik2pRjk2(rij, rik, rjk, rs, df, d2f);
	double g = exp(-eta*f);
	double eg = -eta*exp(-eta*f);

	for(size_t l=0;l<9;l++) firstDeriv[l] = eg*df[l];
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) 
		secondDeriv[l][m] = -eta*eg* df[l]*df[m] + eg*d2f[l][m];

	return g;
}
/* firstDeriv vector with 9 elements : derivatives of f(rij)*f(rik) */
// variables : xi, yi, zi, xj, yj, zj, xk, yk, zk
// get12Derivatives(rij, df,d2f) : xi, yi, zi, xj, yj, zj , df : 6 d2f : 6x6
double getDerivativesfc(const double* rij, const double* rik, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf)
{
	double df6[6];
	double d2f6[6][6];

	double dfij[9];
	double d2fij[9][9];
	double fij = cf.get12Derivatives(rij,  df6, d2f6);
	build9From6ij(df6, dfij);
	build9From6ij(d2f6, d2fij);

	double dfik[9];
	double d2fik[9][9];
	double fik = cf.get12Derivatives(rik,  df6, d2f6);
	build9From6ik(df6, dfik);
	build9From6ik(d2f6, d2fik);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = dfij[l]*fik+dfik[l]*fij;

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] = d2fij[l][m] * fik + dfij[l]*dfik[m] +dfij[m]*dfik[l] + fij*d2fik[l][m];

	return fij*fik;
}
double getDerivativesfc(const double* rij, const double* rik, const double* rjk, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf)
{
	double df6[6];
	double d2f6[6][6];

	double dfij[9];
	double d2fij[9][9];
	double fij = cf.get12Derivatives(rij,  df6, d2f6);
	build9From6ij(df6, dfij);
	build9From6ij(d2f6, d2fij);

	double dfik[9];
	double d2fik[9][9];
	double fik = cf.get12Derivatives(rik,  df6, d2f6);
	build9From6ik(df6, dfik);
	build9From6ik(d2f6, d2fik);

	double dfjk[9];
	double d2fjk[9][9];
	double fjk = cf.get12Derivatives(rjk,  df6, d2f6);
	build9From6jk(df6, dfjk);
	build9From6jk(d2f6, d2fjk);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = dfij[l]*fik*fjk+fij*dfik[l]*fjk+fij*fik*dfjk[l];

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] =  
				  d2fij[l][m]*fik*fjk + fij*d2fik[l][m]*fjk +  fij*fik*d2fjk[l][m]
				+ dfij[l]*(dfik[m]*fjk + fik*dfjk[m])
				+ dfik[l]*(dfij[m]*fjk + fij*dfjk[m])
				+ dfjk[l]*(dfij[m]*fik + fij*dfik[m]);

	return fij*fik*fjk;
}
double getDerivativesG(const double* rij, const double* rik, const double* rjk, double lambda, double zeta, double eta, double rs, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf)
{
	double dft[9];
	double d2ft[9][9];
	double ft = getDerivativescosijk(rij, rik, lambda, zeta, dft, d2ft);
	double dfe[9];
	double d2fe[9][9];
	double fe = getDerivativesexpetaRij2pRik2pRjk2(rij, rik, rjk,  eta, rs, dfe, d2fe);
	double dfc[9];
	double d2fc[9][9];
	double fc = getDerivativesfc(rij, rik, rjk, dfc, d2fc, cf);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = dft[l]*fe*fc+ft*dfe[l]*fc+ft*fe*dfc[l];

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] =  
				  d2ft[l][m]*fe*fc + ft*d2fe[l][m]*fc +  ft*fe*d2fc[l][m]
				+ dft[l]*(dfe[m]*fc + fe*dfc[m])
				+ dfe[l]*(dft[m]*fc + ft*dfc[m])
				+ dfc[l]*(dft[m]*fe + ft*dfe[m]);
	double fact = pow(2.0,1-zeta);
	for(size_t l=0;l<9;l++) firstDeriv[l] *= fact;
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) secondDeriv[l][m] *=fact;  

	return fact*ft*fe*fc;
}
double getDerivativesG(const double* rij, const double* rik, double lambda, double zeta, double eta, double rs, double firstDeriv[], double secondDeriv[][9], const CutoffFunction& cf)
{
	double dft[9];
	double d2ft[9][9];
	double ft = getDerivativescosijk(rij, rik, lambda, zeta, dft, d2ft);

	double dfe[9];
	double d2fe[9][9];
	double fe = getDerivativesexpetaRij2pRik2(rij, rik, eta, rs, dfe, d2fe);
	double dfc[9];
	double d2fc[9][9];
	double fc = getDerivativesfc(rij, rik, dfc, d2fc, cf);

	for(size_t l=0;l<9;l++) 
		firstDeriv[l] = dft[l]*fe*fc+ft*dfe[l]*fc+ft*fe*dfc[l];

	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<9;m++) 
			 secondDeriv[l][m] =  
				  d2ft[l][m]*fe*fc + ft*d2fe[l][m]*fc +  ft*fe*d2fc[l][m]
				+ dft[l]*(dfe[m]*fc + fe*dfc[m])
				+ dfe[l]*(dft[m]*fc + ft*dfc[m])
				+ dfc[l]*(dft[m]*fe + ft*dfe[m]);
	double fact = pow(2.0,1.0-zeta);
	for(size_t l=0;l<9;l++) firstDeriv[l] *= fact;
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) secondDeriv[l][m] *=fact;  

	return fact*ft*fe*fc;
}
void setSecondDerivesijk(double secondDeriv[][9], int indexI, int indexJ, int indexK, double scalingFactor, std::vector< std::vector<double> >& deriv)
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<9;m++) secondDeriv[l][m] *= scalingFactor;
	// atom i & atom i
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexI+ia;
			int indexja = 3*indexI+ja;
			deriv[indexia][indexja] += secondDeriv[ia][ja];
		}
	// atom i & atom j
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexI+ia;
			int indexja = 3*indexJ+ja;
			deriv[indexia][indexja] += secondDeriv[ia][3+ja];
			//deriv[indexja][indexia] += secondDeriv[ia][3+ja];
			deriv[indexja][indexia] += secondDeriv[3+ja][ia];
		}
	// atom j & atom j
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexJ+ia;
			int indexja = 3*indexJ+ja;
			deriv[indexia][indexja] += secondDeriv[3+ia][3+ja];
		}
	// atom i & atom k
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexI+ia;
			int indexja = 3*indexK+ja;
			deriv[indexia][indexja] += secondDeriv[ia][6+ja];
			//deriv[indexja][indexia] += secondDeriv[ia][6+ja];
			deriv[indexja][indexia] += secondDeriv[6+ja][ia];
		}
	// atom j & atom k
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexJ+ia;
			int indexja = 3*indexK+ja;
			deriv[indexia][indexja] += secondDeriv[3+ia][6+ja];
			//deriv[indexja][indexia] += secondDeriv[3+ia][6+ja];
			deriv[indexja][indexia] += secondDeriv[6+ja][3+ia];
		}
	// atom k & atom k
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexK+ia;
			int indexja = 3*indexK+ja;
			deriv[indexia][indexja] += secondDeriv[6+ia][6+ja];
		}
}
void setSecondDerivesij(double secondDeriv[][6], int indexI, int indexJ, double scalingFactor, std::vector< std::vector<double> >& deriv)
{
	for(size_t l=0;l<6;l++) for(size_t m=0;m<6;m++) secondDeriv[l][m] *= scalingFactor;
	// atom i & atom i
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexI+ia;
			int indexja = 3*indexI+ja;
			deriv[indexia][indexja] += secondDeriv[ia][ja];
		}
	// atom i & atom j
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexI+ia;
			int indexja = 3*indexJ+ja;
			deriv[indexia][indexja] += secondDeriv[ia][3+ja];
			//deriv[indexja][indexia] += secondDeriv[ia][3+ja];
			deriv[indexja][indexia] += secondDeriv[3+ja][ia];
		}
	// atom j & atom j
	 for(int ia=0;ia<3;ia++)
		for(int ja=0;ja<3;ja++)
		{
			int indexia = 3*indexJ+ia;
			int indexja = 3*indexJ+ja;
			deriv[indexia][indexja] += secondDeriv[3+ia][3+ja];
		}
}
void toPhysicalUnits(double* deriv, size_t derivSize, double convEnergy, double convLength)
{
	double sFactor = 1.0/convEnergy*convLength*convLength;
	for(size_t l=0;l<derivSize;l++) deriv[l] *= sFactor;
}

}
#endif /* HIGH_DERIVATIVES*/
