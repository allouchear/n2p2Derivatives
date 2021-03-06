// Abdul-Rahman Allouche 2020
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

#ifdef HIGH_DERIVATIVES

#include "Derivatives.h"
#include <cmath>
#include <iostream>


using namespace std;
using namespace nnp;

static void swap(int *a, int *b);
static int getP(int k, int l, int m, int n);
static void getListPermutation(int *arr, int start, int end, int** list, int& il);
static int **getListPermutation(int *arr, int n, int& np);
static int factorial(int n);
static int getIndex(int indexI, int indexJ, int indexK, int ishift);
static int getIndex(int indexI, int indexJ, int ishift);
static double* new1Dtable(int size);
static double* copy1Dtable(int size, double* t);
static void copy1Dtable(int size, double* tc, double* t);
static void free1Dtable(double* t);
static double** new2Dtable(int size);
static double** copy2Dtable(int size, double** t);
static void copy2Dtable(int size, double** tc, double** t);
static void free2Dtable(int size, int** t);
static void free2Dtable(int size, double** t);
static double*** new3Dtable(int size);
static double*** copy3Dtable(int size, double*** t);
static void copy3Dtable(int size, double*** tc, double*** t);
static void free3Dtable(int size, double***t);
static double**** new4Dtable(int size);
static double**** copy4Dtable(int size, double**** t);
static void copy4Dtable(int size, double**** tc, double**** t);
static void free4Dtable(int size, double****t);
static void build9From6ij(double* d6, double* d9);
static void build9From6ik(double* d6, double* d9);
static void build9From6jk(double* d6, double* d9);
static void build9From6ij(double** d6, double** d9);
static void build9From6ik(double** d6, double** d9);
static void build9From6jk(double** d6, double** d9);
static void build9From6ij(double*** d6, double*** d9);
static void build9From6ij(double**** d6, double**** d9);
static void build9From6ik(double*** d6, double*** d9);
static void build9From6ik(double**** d6, double**** d9);
static void build9From6jk(double*** d6, double*** d9);
static void build9From6jk(double**** d6, double**** d9);
static void build9From6(int lm, double* d6, double* d9);
static void build9From6(int lm, double** d6, double** d9);
static void build9From6(int lm, double*** d6, double*** d9);
static void build9From6(int lm, double**** d6, double**** d9);

Derivatives::Derivatives()
{
	maxOrder = 0;
	nVariables = 0;
	value = 0;
	df = NULL;
	d2f = NULL;
	d3f = NULL;
	d4f = NULL;
}
Derivatives::Derivatives(int mOrder)
{
	maxOrder = mOrder;
	nVariables = 0;
	value = 0;
	df = NULL;
	d2f = NULL;
	d3f = NULL;
	d4f = NULL;
}
void Derivatives::free()
{
	if(df) free1Dtable(df);
	if(d2f) free2Dtable(nVariables, d2f);
	if(d3f) free3Dtable(nVariables, d3f);
	if(d4f) free4Dtable(nVariables, d4f);
	maxOrder = 0;
	nVariables = 0;
	value = 0;
	df = NULL;
	d2f = NULL;
	d3f = NULL;
	d4f = NULL;
}
Derivatives::~Derivatives()
{
	free();
}

Derivatives::Derivatives(int mOrder, int nVar)
{
	maxOrder = mOrder;
	nVariables = nVar;
	value = 0;
	df = NULL;
	d2f = NULL;
	d3f = NULL;
	d4f = NULL;
	if(nVariables<1) return;
	if(maxOrder>0) df = new1Dtable(nVariables);
	if(maxOrder>1) d2f = new2Dtable(nVariables);
	if(maxOrder>2) d3f = new3Dtable(nVariables);
	if(maxOrder>3) d4f = new4Dtable(nVariables);
	if(maxOrder>4) cerr<<"maxOrder = "<<maxOrder<<endl;
	if(maxOrder>4) throw std::invalid_argument("ERROR: Derivatives over 4 are not yet implemented Derivatives class.\n");
}
Derivatives::Derivatives(const Derivatives& right)
{
	maxOrder = right.maxOrder;
	nVariables = right.nVariables;
	value = right.value;
	df = NULL;
	d2f = NULL;
	d3f = NULL;
	d4f = NULL;
	if(nVariables<1) return;
	if(right.df) df = copy1Dtable(nVariables, right.df);
	if(right.d2f) d2f = copy2Dtable(nVariables, right.d2f);
	if(right.d3f) d3f = copy3Dtable(nVariables, right.d3f);
	if(right.d4f) d4f = copy4Dtable(nVariables, right.d4f);
	if(maxOrder>4) throw std::invalid_argument("ERROR: Derivatives over 4 are not yet implemented Derivatives class.\n");
}
Derivatives& Derivatives::operator=(const Derivatives& right)
{
	if(&right==this) return *this;
	if(maxOrder == right.maxOrder && nVariables == right.nVariables)
	{
		if(right.df) copy1Dtable(nVariables, right.df, df);
		if(right.d2f) copy2Dtable(nVariables, right.d2f, d2f);
		if(right.d3f) copy3Dtable(nVariables, right.d3f, d3f);
		if(right.d4f) copy4Dtable(nVariables, right.d4f, d4f);
		if(maxOrder>4) throw std::invalid_argument("ERROR: Derivatives over 4 are not yet implemented Derivatives class.\n");
	}
	else
	{
		free();
		maxOrder = right.maxOrder;
		nVariables = right.nVariables;
		df = NULL;
		d2f = NULL;
		d3f = NULL;
		d4f = NULL;
		if(right.df) df = copy1Dtable(nVariables, right.df);
		if(right.d2f) d2f = copy2Dtable(nVariables, right.d2f);
		if(right.d3f) d3f = copy3Dtable(nVariables, right.d3f);
		if(right.d4f) d4f = copy4Dtable(nVariables, right.d4f);
		if(maxOrder>4) throw std::invalid_argument("ERROR: Derivatives over 4 are not yet implemented Derivatives class.\n");
	}
	value = right.value;
	return *this;
}
void Derivatives::resetdf()
{
	if(df)
	for(int l=0;l<nVariables;l++) df[l] = 0;
}
void Derivatives::resetd2f()
{
	if(d2f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			d2f[l][m] = 0;
}
void Derivatives::resetd3f()
{
	if(d3f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				d3f[l][m][n] = 0;
}
void Derivatives::resetd4f()
{
	if(d4f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
					d4f[l][m][n][k] = 0;
}
void Derivatives::reset()
{
	resetdf();
	resetd2f();
	resetd3f();
	resetd4f();
	value = 0;
}
/* size = number of variables 
   first : vector if size elements

   compute derivatives of a*f1+b*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeSumFirst(double a, double b, const Derivatives & f2, Derivatives & fsum) const
{
	if(df)
	for(int l=0;l<nVariables;l++) 
		fsum.df[l] = a*df[l]+b*f2.df[l];
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 

   compute derivatives of a*f1+b*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeSumSecond(double a, double b, const Derivatives & f2, Derivatives & fsum) const
{
	if(d2f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			fsum.d2f[l][m] = a*d2f[l][m] + b*f2.d2f[l][m];
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 
   third : size*size*size

   compute derivatives of a*f1+b*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeSumThird(double a, double b, const Derivatives & f2, Derivatives & fsum) const
{
	if(d3f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
			{
				double v = a*d3f[l][m][n] + b*f2.d3f[l][m][n];
				fsum.d3f[l][m][n] = v;
			}
}
/* nVariables = number of variables 
   first : vector if nVariables elements
   second : matrix nVariables*nVariables 
   third : nVariables*nVariables*nVariables
   fourth : nVariables*nVariables*nVariables

   compute derivatives of a*f1+b*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeSumFourth(double a, double b, const Derivatives & f2, Derivatives & fsum) const
{
	if(d4f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
				{
					double v = a*d4f[l][m][n][k] + b*f2.d4f[l][m][n][k];
					fsum.d4f[l][m][n][k] = v;
				}
}
void Derivatives::computeSum(double a, double b, const Derivatives & f2, Derivatives & fsum) const
{
	if(nVariables != f2.nVariables || maxOrder != f2.maxOrder ) 
		throw std::invalid_argument("ERROR: I cannnot add 2 Derivaritves  : f1 &f2 have not the sam number of variables in Derivatives class.\n");
	
	if(nVariables != fsum.nVariables || maxOrder != fsum.maxOrder ) 
	{
		fsum.free();
		fsum = Derivatives(maxOrder, nVariables);
	}
	computeSumFirst( a,  b, f2, fsum);
	computeSumSecond( a,  b, f2, fsum);
	computeSumThird( a,  b, f2, fsum);
	computeSumFourth( a,  b, f2, fsum);
	fsum.value = a*value + b*f2.value;
}
void Derivatives::computeSum(const Derivatives & f2, Derivatives & fsum) const
{
	computeSum(1.0, 1.0, f2, fsum);
}
Derivatives Derivatives::computeSum(double a, double b, const Derivatives & f2) const
{
	if(nVariables != f2.nVariables || maxOrder != f2.maxOrder ) 
		throw std::invalid_argument("ERROR: I cannnot add 2 Derivaritves  : f1 &f2 have not the sam number of variables in Derivatives class.\n");
	
	Derivatives fsum(maxOrder, nVariables);
	computeSum(a, b, f2, fsum);
	return fsum;
}
Derivatives Derivatives::computeSum(const Derivatives & f2) const
{
	return computeSum(1.0,1.0,f2);
}
Derivatives Derivatives::operator+(const Derivatives & f2) const
{
	return computeSum(1.0,1.0,f2);
}
void Derivatives::operator+=(const Derivatives & f2)
{
	*this = *this + f2;
}
void Derivatives::operator+=(double v)
{
	// no change in derivative
	value += v;
}
/* size = number of variables 
   first : vector if size elements

   compute derivatives of f1*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeProdFirst(const Derivatives& f2, Derivatives& fprod) const
{
	if(df)
	for(int l=0;l<nVariables;l++) 
		fprod.df[l] = df[l]*f2.value+value*f2.df[l];
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 

   compute derivatives of f1*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeProdSecond(const Derivatives& f2, Derivatives& fprod) const
{
	if(d2f && df)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			fprod.d2f[l][m] = d2f[l][m]*f2.value + value*f2.d2f[l][m] + df[l]*f2.df[m]+df[m]*f2.df[l];
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 
   third : size*size*size

   compute derivatives of f1*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeProdThird(const Derivatives& f2, Derivatives& fprod) const
{
	if(d3f && d2f && df)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
			{
				double v = 
				  d3f[l][m][n] *f2.value + value*f2.d3f[l][m][n]
				+ d2f[l][m]*f2.df[n]+ d2f[l][n]*f2.df[m]+ d2f[m][n]*f2.df[l]+
				+ f2.d2f[l][m]*df[n]+ f2.d2f[l][n]*df[m]+ f2.d2f[m][n]*df[l];
				fprod.d3f[l][m][n] = v;
			}
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 
   third : size*size*size
   fourth : size*size*size

   compute derivatives of f1*f2 using derivatives of f1 and those of f2 
*/
void Derivatives::computeProdFourth(const Derivatives& f2, Derivatives& fprod) const
{
	if(d4f && d3f && d2f && df)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
				{
					double v = 
				  	d4f[l][m][n][k] *f2.value + value*f2.d4f[l][m][n][k]
					+ d3f[l][m][n]*f2.df[k]+ d3f[l][m][k]*f2.df[n]+ d3f[l][n][k]*f2.df[m] + d3f[m][n][k]*f2.df[l]
					+ f2.d3f[l][m][n]*df[k]+ f2.d3f[l][m][k]*df[n]+ f2.d3f[l][n][k]*df[m] + f2.d3f[m][n][k]*df[l] 

					+ d2f[l][m]*f2.d2f[n][k]+ d2f[l][n]*f2.d2f[m][k]+ d2f[l][k]*f2.d2f[m][n]
					+ f2.d2f[l][m]*d2f[n][k]+ f2.d2f[l][n]*d2f[m][k]+ f2.d2f[l][k]*d2f[m][n];

					fprod.d4f[l][m][n][k] = v; 
			}
}
void Derivatives::computeProd(const Derivatives& f2, Derivatives& fprod) const
{
	if(nVariables != f2.nVariables || maxOrder != f2.maxOrder ) 
		throw std::invalid_argument("ERROR: I cannnot do a prod of 2 Derivaritves  : f1 &f2 have not the sam number of variables in Derivatives class.\n");
	
	if(nVariables != fprod.nVariables || maxOrder != fprod.maxOrder ) 
	{
		fprod.free();
		fprod = Derivatives(maxOrder, nVariables);
	}
	computeProdFirst(f2, fprod);
	computeProdSecond(f2, fprod);
	computeProdThird(f2, fprod);
	computeProdFourth(f2, fprod);
	fprod.value = value * f2.value;
}
Derivatives Derivatives::computeProd(const Derivatives& f2) const
{
	 if(nVariables != f2.nVariables || maxOrder != f2.maxOrder )
		throw std::invalid_argument("ERROR: I cannnot do a prod of 2 Derivaritves  : f1 &f2 have not the same number of variables in Derivatives class.\n");

        Derivatives fprod(maxOrder, nVariables);
        computeProd(f2, fprod);
	return fprod;

}
Derivatives Derivatives::operator*(const Derivatives& f2) const
{
	 if(nVariables != f2.nVariables || maxOrder != f2.maxOrder )
		throw std::invalid_argument("ERROR: I cannnot do a prod of 2 Derivaritves  : f1 &f2 have not the same number of variables in Derivatives class.\n");

        Derivatives fprod(maxOrder, nVariables);
        computeProd(f2, fprod);
	return fprod;
}
void Derivatives::operator*=(const Derivatives& f2)
{
	*this = *this * f2;
}
/* size = number of variables 
   first : vector if size elements

   compute derivatives of s*f1 using derivatives of f1
*/
void Derivatives::computeProdFirst(double s, Derivatives& fprod) const
{
	if(df)
	for(int l=0;l<nVariables;l++) 
		fprod.df[l] = df[l]*s;
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 

   compute derivatives of v*f1 using derivatives of f1
*/
void Derivatives::computeProdSecond(double s, Derivatives& fprod) const
{
	if(d2f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			fprod.d2f[l][m] = d2f[l][m]*s;
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 
   third : size*size*size

   compute derivatives of v*f1 using derivatives of f1
*/
void Derivatives::computeProdThird(double s, Derivatives& fprod) const
{
	if(d3f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				fprod.d3f[l][m][n] = d3f[l][m][n] *s;
}
/* size = number of variables 
   first : vector if size elements
   second : matrix size*size 
   third : size*size*size
   fourth : size*size*size

   compute derivatives of v*f1 using derivatives of f1
*/
void Derivatives::computeProdFourth(double s, Derivatives& fprod) const
{
	if(d4f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
				  	fprod.d4f[l][m][n][k]  = d4f[l][m][n][k] *s;
}
void Derivatives::computeProd(double s, Derivatives& fprod) const
{
	if(nVariables != fprod.nVariables || maxOrder != fprod.maxOrder ) 
	{
		fprod.free();
		fprod = Derivatives(maxOrder, nVariables);
	}
	computeProdFirst(s, fprod);
	computeProdSecond(s, fprod);
	computeProdThird(s, fprod);
	computeProdFourth(s, fprod);
	fprod.value = value * s;
}
Derivatives Derivatives::computeProd(double s) const
{
        Derivatives fprod(maxOrder, nVariables);
        computeProd(s, fprod);
	return fprod;

}
Derivatives Derivatives::operator*(double s) const
{
        Derivatives fprod(maxOrder, nVariables);
        computeProd(s, fprod);
	return fprod;
}
void Derivatives::operator*=(double s)
{
	*this = *this * s;
}
/*
   compute derivatives of f1*f2*f3 using derivatives of f1, f2 and those of f3
*/
void Derivatives::computeProd(const Derivatives& f2, const Derivatives& f3, Derivatives& fprod)
{
	Derivatives fprod12  = computeProd(f2);
	fprod  = fprod12.computeProd(f3);
}
Derivatives Derivatives::computeProd(const Derivatives& f2, const Derivatives& f3)
{
	Derivatives fprod12  = computeProd(f2);
	Derivatives fprod123  = fprod12.computeProd(f3);
	return fprod123;
}
/* 
  compute derivatives of f(u), using df/du (dnf) and du/dxi (nVariables variable)
   dnf[0] = f(u)
   dnf[1] = df(u)/du
   dnf[n] = dnf(u)/du^n
*/
void Derivatives::computedfu(const double* dnf, const Derivatives& u)
{
	if(df)
	for(int l=0;l<nVariables;l++) 
		df[l] = dnf[1]*u.df[l];
}
void Derivatives::computed2fu(const double* dnf, const Derivatives& u)
{
	if(d2f)
        for(int l=0;l<nVariables;l++)
        	for(int m=0;m<=l;m++)
			d2f[l][m] = u.df[l]*u.df[m]*dnf[2] + u.d2f[l][m]*dnf[1];
}
void Derivatives::computed3fu(const double* dnf, const Derivatives& u)
{
	if(d3f)
        for(int l=0;l<nVariables;l++)
        	for(int m=0;m<=l;m++)   
        		for(int n=0;n<=m;n++)   
			{
				double v = 
					  dnf[3]*(
						u.df[l]*u.df[m]*u.df[n]
						)
					+ dnf[2]*(
						u.d2f[l][m]*u.df[n] +
						u.d2f[l][n]*u.df[m] +
						u.d2f[m][n]*u.df[l] 
						)
					+ dnf[1]*(
						u.d3f[l][m][n]
						);
                                d3f[l][m][n] = v;
			}
}
void Derivatives::computed4fu(const double* dnf, const Derivatives& u)
{
	if(d4f)
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
				{
					double v = 
				  	  dnf[4]*(u.df[l]*u.df[m]*u.df[n]*u.df[k])
					+ dnf[3]*(
					u.d2f[l][m]*u.df[n]*u.df[k]+ 
					u.d2f[l][n]*u.df[m]*u.df[k]+
					u.d2f[l][k]*u.df[m]*u.df[n]+
					u.d2f[m][n]*u.df[l]*u.df[k]+
					u.d2f[m][k]*u.df[l]*u.df[n]+
					u.d2f[n][k]*u.df[l]*u.df[m]
					)
					+ dnf[2]*(
					u.d3f[l][m][n]*u.df[k]+
					u.d3f[l][m][k]*u.df[n]+
					u.d3f[l][n][k]*u.df[m]+
					u.d3f[m][n][k]*u.df[l]+
					u.d2f[l][m]*u.d2f[n][k]+
					u.d2f[l][n]*u.d2f[m][k]+
					u.d2f[l][k]*u.d2f[m][n]
					)
					+ dnf[1]*(u.d4f[l][m][n][k]);

					d4f[l][m][n][k] = v;
			}
}
void Derivatives::computednfu(const double* dnf, const Derivatives& u)
{
	if(nVariables != u.nVariables || maxOrder != u.maxOrder ) 
	{
		free();
		*this = Derivatives(u.maxOrder, u.nVariables);
	}
	computedfu(dnf, u);
	computed2fu(dnf, u);
	computed3fu(dnf, u);
	computed4fu(dnf, u);
	value = dnf[0];
}
void Derivatives::computednexpu(const Derivatives& u)
{
	double* dnf = NULL;
	dnf = new double[maxOrder+1];
	double d=exp(u.value);
	for(int i=0;i<=maxOrder;i++) dnf[i] = d;
	computednfu(dnf,u);
	if(dnf) delete [] dnf;
}
/* nVariables = number of variables 
   df : vector if nVariables elements

   compute derivatives of F^zeta using derivatives of F
*/
void Derivatives::computedFzeta(const Derivatives& F, double zeta)
{
	double Fz = pow(F.value,zeta);
	double overF = (abs(F.value>1e-14))?1.0/F.value:0;
	double Fzm1 = zeta*Fz*overF;
	for(int l=0;l<nVariables;l++) 
		df[l] = Fzm1*F.df[l];
	value = Fz;
}
/* nVariables = number of variables 
   df : vector if nVariables elements
   d2f : matrix nVariables*nVariables 

   compute derivatives of f^zeta using derivatives of F
*/
void Derivatives::computed2Fzeta(const Derivatives& F, double zeta)
{
	double Fz = pow(F.value,zeta);
	double overF = (abs(F.value>1e-14))?1.0/F.value:0;
	double Fzm1 = zeta*Fz*overF;
	double Fzm2 = (zeta-1)*Fzm1*overF;
	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			d2f[l][m] = F.d2f[l][m]*Fzm1 + Fzm2*F.df[l]*F.df[m];
	value = Fz;

}
/* nVariables = number of variables 
   df : vector if nVariables elements
   d2f : matrix nVariables*nVariables 
   d3f : nVariables*nVariables*nVariables

   compute derivatives of f^zeta using derivatives of F
*/
void Derivatives::computed3Fzeta(const Derivatives& F, double zeta)
{
	double Fz = pow(F.value,zeta);
	double overF = (abs(F.value>1e-14))?1.0/F.value:0;
	double Fzm1 = zeta*Fz*overF;
	double Fzm2 = (zeta-1)*Fzm1*overF;
	double Fzm3 = (zeta-2)*Fzm2*overF;

	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
			{
				double v = 
				  Fzm3*(F.df[l]*F.df[m]*F.df[n])
				+ Fzm2*(F.d2f[l][m]*F.df[n]+F.d2f[l][n]*F.df[m]+F.d2f[m][n]*F.df[l]) 
				+ Fzm1*(F.d3f[l][m][n]);
				d3f[l][m][n] = v;
			}
	value = Fz;
}
/* nVariables = number of variables 
   df : vector if nVariables elements
   d2f : matrix nVariables*nVariables 
   d3f : nVariables*nVariables*nVariables
   d4f : nVariables*nVariables*nVariables

   compute derivatives of f^zeta using derivatives of F
*/
void Derivatives::computed4Fzeta(const Derivatives& F, double zeta)
{
	double Fz = pow(F.value,zeta);
	double overF = (abs(F.value>1e-14))?1.0/F.value:0;
	double Fzm1 = zeta*Fz*overF;
	double Fzm2 = (zeta-1)*Fzm1*overF;
	double Fzm3 = (zeta-2)*Fzm2*overF;
	double Fzm4 = (zeta-3)*Fzm3*overF;

	for(int l=0;l<nVariables;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
				{
					double v = 
				  	  Fzm4*(F.df[l]*F.df[m]*F.df[n]*F.df[k])
					+ Fzm3*(
					F.d2f[l][m]*F.df[n]*F.df[k]+ 
					F.d2f[l][n]*F.df[m]*F.df[k]+
					F.d2f[l][k]*F.df[m]*F.df[n]+
					F.d2f[m][n]*F.df[l]*F.df[k]+
					F.d2f[m][k]*F.df[l]*F.df[n]+
					F.d2f[n][k]*F.df[l]*F.df[m]
					)
					+ Fzm2*(
					F.d3f[l][m][n]*F.df[k]+
					F.d3f[l][m][k]*F.df[n]+
					F.d3f[l][n][k]*F.df[m]+
					F.d3f[m][n][k]*F.df[l]+
					F.d2f[l][m]*F.d2f[n][k]+
					F.d2f[l][n]*F.d2f[m][k]+
					F.d2f[l][k]*F.d2f[m][n]
					)
					+ Fzm1*(F.d4f[l][m][n][k]);

					d4f[l][m][n][k] = v;
			}
	value = Fz;
}
void Derivatives::computednFzeta(const Derivatives& F, double zeta)
{
	if(nVariables != F.nVariables || maxOrder != F.maxOrder ) 
	{
		free();
		*this = Derivatives(F.maxOrder, F.nVariables);
	}
	if(df) computedFzeta(F,zeta);
	if(d2f) computed2Fzeta(F,zeta);
	if(d3f) computed3Fzeta(F,zeta);
	if(d4f) computed4Fzeta(F,zeta);
	if(maxOrder<1) value = pow(F.value,zeta);
}
Derivatives Derivatives::getdnFzeta(double zeta) const
{
	Derivatives dnFz;
	dnFz.computednFzeta(*this, zeta);
	return dnFz;
}
/* firstDeriv vector with 9 elements */
/* Derivatives of Rij dot Rik*/
void Derivatives::computedRijk(const double* rij, const double* rik)
{
	// rij[l] = ri[l]-rj[l]
	for(size_t l=0;l<3;l++) df[l] = rij[l]+rik[l];// d/dxi, d/dyi, d/dzi
	for(size_t l=3;l<6;l++) df[l] = -rik[l-3];// d/dxj, d/dyj, d/dzj
	for(size_t l=6;l<9;l++) df[l] = -rij[l-6];// d/dxk, d/dyk, d/dzk
}
/* second matrix with 9x9 elements */
/* first vector with 9 elements */
void Derivatives::computed2Rijk()
{
	for(size_t l=0;l<9;l++)  
		for(size_t m=0;m<=l;m++)
			d2f[l][m] = 0;

	for(size_t l=0;l<3;l++)  d2f[l][l] = 2; // d/dxi2, d/dyi2, d/dzi2
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
	for(size_t l=0;l<3;l++)  d2f[l+3][l] = d2f[l+6][l] = -1;
	for(size_t l=0;l<3;l++)  d2f[l+6][l+3] = 1;
}
/* third 9x9x9 elements */
/* second matrix with 9x9 elements */
/* first vector with 9 elements */
void Derivatives::computed3Rijk()
{
	for(size_t l=0;l<9;l++)  
		for(size_t m=0;m<=l;m++)
			for(size_t n=0;n<=m;n++)
				d3f[l][m][n] = 0;
}
/* fourth 9x9x9x9 elements */
/* third 9x9x9 elements */
/* second matrix with 9x9 elements */
/* first vector with 9 elements */
void Derivatives::computed4Rijk()
{
	for(size_t l=0;l<9;l++)  
		for(size_t m=0;m<=l;m++)
			for(size_t n=0;n<=m;n++)
				for(size_t k=0;k<=n;k++)
					d4f[l][m][n][k] = 0;
}
void Derivatives::computednRijk(const double* rij, const double* rik)
{
	if(nVariables != 9) 
	{
		int mOrder = maxOrder;
		free();
		*this = Derivatives(mOrder, 9);
	}
	if(df) computedRijk(rij,rik);
	if(d2f) computed2Rijk();
	if(d3f) computed3Rijk();
	if(d4f) computed4Rijk();
	value = 0;
	for(size_t l=0;l<3;l++) value += rij[l]*rik[l];// rij dot rik
}
Derivatives::Derivatives(int mOrder, const double* rij, const double* rik): Derivatives(mOrder, 9)
{
	computednRijk(rij, rik);
}
/* df  : vector with 6 elements
   derivative of Rij2 = (xi-xj)^2+ (zi-zj)^2+(zi-zj)^2
   6 variables : xi, yi, zi, xj,yj,zj
*/
void Derivatives::computedRij2_6(const double* rij)
{
	for(size_t l=0;l<3;l++) df[l] = 2*rij[l];
	for(size_t l=3;l<6;l++) df[l] = -df[l-3];
}
/* df  : vector with 6 elements
 d2f : matrix with 6x6 elements
  derivative of Rij2 = (xi-xj)^2+ (zi-zj)^2+(zi-zj)^2
  6 variables : xi, yi, zi, xj,yj,zj
*/
void Derivatives::computed2Rij2_6()
{
	for(size_t l=0;l<6;l++)
		for(size_t m=0;m<=l;m++)
				d2f[l][m] = 0;

	for(size_t l=0;l<3;l++)   d2f[l][l] = 2;
	for(size_t l=3;l<6;l++)   d2f[l][l] = 2;
	for(size_t l=0;l<3;l++)   d2f[l+3][l]  = -2;
}

/* df  : vector with 6 elements
 d2f : matrix with 6x6 elements
 Third  : 6x6x6 elements 
  derivative of Rij2 = (xi-xj)^2+ (zi-zj)^2+(zi-zj)^2
  6 variables : xi, yi, zi, xj,yj,zj
*/
void Derivatives::computed3Rij2_6()
{
	for(size_t l=0;l<6;l++)
		for(size_t m=0;m<=l;m++)
			for(size_t n=0;n<=m;n++)
				d3f[l][m][n] = 0;
}
/* df  : vector with 6 elements
 d2f : matrix with 6x6 elements
 Third  : 6x6x6 elements 
  derivative of Rij2 = (xi-xj)^2+ (zi-zj)^2+(zi-zj)^2
  6 variables : xi, yi, zi, xj,yj,zj
*/
void Derivatives::computed4Rij2_6()
{
	for(size_t l=0;l<6;l++)
		for(size_t m=0;m<=l;m++)
			for(size_t n=0;n<=m;n++)
				for(size_t k=0;k<=n;k++)
					d4f[l][m][n][k] = 0;
}
void Derivatives::computednRij2_6(const double* rij)
{
	if(nVariables != 6) 
	{
		int mOrder = maxOrder;
		free();
		*this = Derivatives(mOrder, 6);
	}
	if(df) computedRij2_6(rij);
	if(d2f) computed2Rij2_6();
	if(d3f) computed3Rij2_6();
	if(d4f) computed4Rij2_6();
	value = 0;
	for(size_t l=0;l<3;l++) value += rij[l]*rij[l];
}
void Derivatives::build9VFrom6V(int lm, Derivatives& d6)
{
	if(d6.nVariables != 6) 
		throw std::invalid_argument("ERROR: Error in Derivatives:::build9VFrom6V.\n");
	if(nVariables != 9) 
	{
		int mOrder = maxOrder;
		free();
		*this = Derivatives(mOrder, 9);
	}
	if(df) build9From6(lm, d6.df, df);
	if(d2f) build9From6(lm, d6.d2f, d2f);
	if(d3f) build9From6(lm, d6.d3f, d3f);
	if(d4f) build9From6(lm, d6.d4f, d4f);
	value = d6.value;
}
void Derivatives::computednR2_9(int lm, const double* r)
{
	Derivatives d6(maxOrder);
	d6.computednRij2_6(r);
	build9VFrom6V(lm, d6);
}
void Derivatives::computednR2(int lm, const double* r)
{
	computednR2_9(lm,r);
}
Derivatives::Derivatives(int mOrder, int lm, const double* r): Derivatives(mOrder, 9)
{
	computednR2(lm,  r);
}
// derivatives of cos theatijk
void Derivatives::computecosijk(const double* rij, const double* rik)
{
	// derivatives of rij^2
	Derivatives f(maxOrder, 0, rij);

	// derivatives of rik^2
	Derivatives g(maxOrder, 1, rik);

	// derivatives of (rij^2*rik^2)^-2
	// in current object
	computednFzeta(f*g,-0.5);

	g.free();
	// derivatives of  Rij dot Rik 
	f.computednRijk(rij, rik);
	// derivatives of Rij dot Rik *(rij^2*rik^2)^-2
	*this = *this*f;
}
// derivatives of (1+lambda cos theatijk)^zeta
void Derivatives::computecosijk(double zeta, double lambda, const double* rij, const double* rik)
{
	Derivatives dncos(maxOrder, nVariables);
	// dnCos
	dncos.computecosijk (rij, rik);

	//F = lambda cos
	dncos *= lambda;
	//F = 1 + lambda cos
	dncos.value += 1.0;
	// (1+lambda cos)^zeta
	computednFzeta(dncos,zeta);
}
// compute drivative of fc(u) with u = rij (lm=0) or rik (lm=1) or rjk (lm=2)
void Derivatives::computednfc(const CutoffFunction& cf, int lm, const double* r)
{
	// derivative of r2
	Derivatives r2(maxOrder, lm,r);
        // derivatives of (r^2)^1/2
	Derivatives u(maxOrder, nVariables);
        u.computednFzeta(r2, 0.5);

	double* dnf = NULL;
	if(maxOrder>0)dnf = new double[maxOrder+1];
	for(int i=0;i<=maxOrder;i++) dnf[i] = cf.dnf(u.value,i);

	computednfu(dnf, u);
	if(dnf) delete [] dnf;
}
// compute drivative of fc(rij) * fc(ijk)
void Derivatives::computednfc(const CutoffFunction& cf, const double* rij, const double* rik)
{
	Derivatives fcij(maxOrder, nVariables);
	fcij.computednfc(cf, 0, rij);
	Derivatives fcik(maxOrder, nVariables);
	fcik.computednfc(cf, 1, rik);
	*this = fcij*fcik;
}
// compute drivative of fc(rij) * fc(rjk) *fc(rjk)
void Derivatives::computednfc(const CutoffFunction& cf, const double* rij, const double* rik, const double* rjk)
{
	Derivatives fcij(maxOrder, nVariables);
	fcij.computednfc(cf, 0, rij);
	Derivatives fcik(maxOrder, nVariables);
	fcik.computednfc(cf, 1, rik);
	Derivatives fcjk(maxOrder, nVariables);
	fcjk.computednfc(cf, 2, rjk);
	*this = fcij*fcik*fcjk;
}
// compute drivative of aij * Rij^2 + aik Rik^2
void Derivatives::computednRij2pRik2(double aij, double aik, const double* rij, const double* rik)
{
	Derivatives rij2(maxOrder, 0,rij);
	Derivatives rik2(maxOrder, 1,rik);
	*this = rij2.computeSum(aij, aik, rik2);
}
// compute drivative of exp(-eta*(Rij^2 + Rik^2)
void Derivatives:: computednexpmetaRij2pRik2(double eta, const double* rij, const double* rik)
{
	Derivatives u(maxOrder, nVariables);
	u.computednRij2pRik2(-eta, -eta , rij, rik);
	computednexpu(u);
}
// compute drivative of aij * Rij^2 + aik Rik^2 + ajk Rjk^2
void Derivatives::computednRij2pRik2pRjk2(double aij, double aik, double ajk, const double* rij, const double* rik, const double* rjk)
{
	Derivatives rij2(maxOrder, 0,rij);
	Derivatives rik2(maxOrder, 1,rik);
	Derivatives rjk2(maxOrder, 2,rjk);
	*this = rij2.computeSum(aij, aik, rik2);
	*this = computeSum(1.0, ajk, rjk2);
}
// compute drivative of exp(-eta*(Rij^2 + Rik^2 + Rijk^2)
void Derivatives:: computednexpmetaRij2pRik2pRjk2(double eta, const double* rij, const double* rik, const double* rjk)
{
	Derivatives u(maxOrder, nVariables);
	u.computednRij2pRik2pRjk2(-eta, -eta , -eta, rij, rik, rjk);
	computednexpu(u);
}
// compute drivative of exp(-eta*(R^2)
void Derivatives:: computednexpmetaR2(int lm, double eta, const double* r)
{
	Derivatives r2(maxOrder, lm,r);
	r2 *= -eta;
	computednexpu(r2);
}
// (r-rs)^2 = r2 - 2 rs r + rs^2
Derivatives::Derivatives(int mOrder, int lm, const double* r, double rs): Derivatives(mOrder, 9)
{
	// *this = r2
	Derivatives r2(maxOrder, lm,  r);
	// r2^0.5
	Derivatives r1 = r2.getdnFzeta(0.5);
	*this = r2 + r1*(-2.0*rs);
	*this += rs*rs;
}
// compute drivative of aij * (Rij-rs)^2 + aik (Rik-rs)^2
void Derivatives::computednRij2pRik2(double aij, double aik, const double* rij, const double* rik, double rs)
{
	Derivatives rij2(maxOrder, 0,rij, rs);
	Derivatives rik2(maxOrder, 1,rik, rs);
	*this = rij2.computeSum(aij, aik, rik2);
}
// compute drivative of exp(-eta*((Rij-rs)^2 + (Rik-rs)^2)
void Derivatives:: computednexpmetaRij2pRik2(double eta, const double* rij, const double* rik, double rs)
{
	Derivatives u(maxOrder, nVariables);
	u.computednRij2pRik2(-eta, -eta , rij, rik, rs);
	computednexpu(u);
}
// compute drivative of aij * (Rij-rs)^2 + aik (Rik-rs)^2 + ajk (Rjk-rs)^2
void Derivatives::computednRij2pRik2pRjk2(double aij, double aik, double ajk, const double* rij, const double* rik, const double* rjk, double rs)
{
	Derivatives rij2(maxOrder, 0,rij, rs);
	Derivatives rik2(maxOrder, 1,rik, rs);
	Derivatives rjk2(maxOrder, 2,rjk, rs);
	*this = rij2.computeSum(aij, aik, rik2);
	*this = computeSum(1.0, ajk, rjk2);
}
// compute drivative of exp(-eta*((Rij-rs)^2 + (Rik-rs)^2 + (Rijk-rs)^2)
void Derivatives:: computednexpmetaRij2pRik2pRjk2(double eta, const double* rij, const double* rik, const double* rjk, double rs)
{
	Derivatives u(maxOrder, nVariables);
	u.computednRij2pRik2pRjk2(-eta, -eta , -eta, rij, rik, rjk, rs);
	computednexpu(u);
}
// compute drivative of exp(-eta*((R-rs)^2)
void Derivatives:: computednexpmetaR2(int lm, double eta, const double* r, double rs)
{
	Derivatives r2(maxOrder, lm, r, rs);
	r2 *= -eta;
	computednexpu(r2);
}
/*  derivatives of exp(-etat(R-rs)^2) *fc(r) */
void Derivatives::computeG(int lm, double eta, const double* r, double rs, const CutoffFunction& fc)
{
	Derivatives er2(maxOrder, 9);
	er2.computednexpmetaR2(lm, eta, r, rs);
	Derivatives fcr(maxOrder,9);
	
	fcr.computednfc(fc, lm, r);
	*this = er2*fcr;
}
/*  derivatives of exp(-etat(Rij-rs)^2 *fc(Rij)*/
void Derivatives::computeG(double eta, const double* rij, double rs, const CutoffFunction& fc)
{
	computeG(0, eta,  rij, rs, fc);
}
/*  derivatives of exp(-etat(R-rs)^2 *fc(R)*/
Derivatives::Derivatives(int maxOrder, int lm, double eta, const double* r, double rs, const CutoffFunction& fc):Derivatives(maxOrder,9)
{
	computeG(lm, eta,  r, rs, fc);
}
/*  derivatives of exp(-etat(Rij-rs)^2 *fc(Rij) */
Derivatives::Derivatives(int maxOrder, double eta, const double* rij, double rs, const CutoffFunction& fc):Derivatives(maxOrder,9)
{
	computeG(0, eta,  rij, rs, fc);
}
/*  derivatives of 
	2^(1-zeta) 
	Rij*Rik/|Rij||Rik|
	*exp(-etat((Rij-rs)^2+(Rik-rs)^2 +(Rjk-rs)^2))
	*fc(Rij)*fc(Rik)*fc(Rjk)
*/
void Derivatives::computeG(double zeta, double lambda, double eta, const double* rij, const double* rik, const double* rjk, double rs, const CutoffFunction& cf)
{
	// dcos
	Derivatives A(maxOrder, 9);
	A.computecosijk(zeta, lambda, rij, rik);
	*this = A;

	// dr2
	A.computednexpmetaRij2pRik2pRjk2(eta, rij, rik, rjk, rs);
	*this *= A;

	//fcr 
	A.computednfc(cf, rij, rik, rjk);
	*this *= A;

	*this *= pow(2.0,1-zeta);
}
/*  derivatives of 
	2^(1-zeta) 
	Rij*Rik/|Rij||Rik|
	*exp(-etat((Rij-rs)^2+(Rik-rs)^2 +(Rjk-rs)^2))
	*fc(Rij)*fc(Rik)*fc(Rjk)
*/
Derivatives::Derivatives(int maxOrder, double zeta, double lambda, double eta, const double* rij, const double* rik, const double* rjk, double rs, const CutoffFunction& cf) :Derivatives(maxOrder,9)
{
	computeG(zeta, lambda, eta, rij, rik, rjk, rs, cf);
}
/*  derivatives of 
	2^(1-zeta) 
	Rij*Rik/|Rij||Rik|
	*exp(-etat((Rij-rs)^2+(Rik-rs)^2))
	*fc(Rij)*fc(Rik)
*/
void Derivatives::computeG(double zeta, double lambda, double eta, const double* rij, const double* rik, double rs, const CutoffFunction& cf)
{
	Derivatives dcos(maxOrder, 9);
	dcos.computecosijk(zeta, lambda, rij, rik);



	Derivatives dr2(maxOrder, 9);
	dr2.computednexpmetaRij2pRik2(eta, rij, rik, rs);

	Derivatives fcr(maxOrder,9);
	fcr.computednfc(cf, rij, rik);

	*this = dcos*dr2*fcr;
	*this *= pow(2.0,1-zeta);
}
/*  derivatives of 
	2^(1-zeta) 
	Rij*Rik/|Rij||Rik|
	*exp(-etat((Rij-rs)^2+(Rik-rs)^2))
	*fc(Rij)*fc(Rik)*
*/
Derivatives::Derivatives(int maxOrder, double zeta, double lambda, double eta, const double* rij, const double* rik, double rs, const CutoffFunction& cf)
:Derivatives(maxOrder,9)
{
	computeG(zeta, lambda, eta, rij, rik, rs, cf);
}


static int getIndex(int indexI, int indexJ, int indexK, int ishift)
{
	int index = 3*indexI;
	if(ishift==3) index = 3*indexJ;
	if(ishift==6) index = 3*indexK;
	return index;
}
static int getIndex(int indexI, int indexJ, int ishift)
{
	int index = 3*indexI;
	if(ishift==3) index = 3*indexJ;
	return index;
}
void Derivatives::addDijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=6;iashift+=3)
	for(int ia=0;ia<3;ia++)
	{
		int i = getIndex(indexI, indexJ, indexK, iashift) + ia;
		v = scalingFactor*deriv9.df[ia+iashift];
		df[i] += v;
	}
}

void Derivatives::addD2ijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=6;iashift+=3)
	for(int jashift=0;jashift<=6;jashift+=3)
	for(int ia=0;ia<3;ia++)
	for(int ja=0;ja<3;ja++)
	{
		int i = getIndex(indexI, indexJ, indexK, iashift) + ia;
		int j = getIndex(indexI, indexJ, indexK, jashift) + ja;
		if(j>i) continue;
		v = scalingFactor*deriv9(ia+iashift,ja+jashift);
		d2f[i][j] += v;
	}
}
void Derivatives::addD3ijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=6;iashift+=3)
	for(int jashift=0;jashift<=6;jashift+=3)
	for(int kashift=0;kashift<=6;kashift+=3)
	for(int ia=0;ia<3;ia++)
	for(int ja=0;ja<3;ja++)
	for(int ka=0;ka<3;ka++)
	{
		int i = getIndex(indexI, indexJ, indexK, iashift) + ia;
		int j = getIndex(indexI, indexJ, indexK, jashift) + ja;
		if(j>i) continue;
		int k = getIndex(indexI, indexJ, indexK, kashift) + ka;
		if(k>j) continue;
		v = scalingFactor*deriv9(ia+iashift,ja+jashift,ka+kashift);
		d3f[i][j][k] += v;
	}
}
void Derivatives::addD4ijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=6;iashift+=3)
	for(int jashift=0;jashift<=6;jashift+=3)
	for(int kashift=0;kashift<=6;kashift+=3)
	for(int lashift=0;lashift<=6;lashift+=3)
	for(int ia=0;ia<3;ia++)
	for(int ja=0;ja<3;ja++)
	for(int ka=0;ka<3;ka++)
	for(int la=0;la<3;la++)
	{
		int i = getIndex(indexI, indexJ, indexK, iashift) + ia;
		int j = getIndex(indexI, indexJ, indexK, jashift) + ja;
		if(j>i) continue;
		int k = getIndex(indexI, indexJ, indexK, kashift) + ka;
		if(k>j) continue;
		int l = getIndex(indexI, indexJ, indexK, lashift) + la;
		if(l>k) continue;
		v = scalingFactor*deriv9(ia+iashift,ja+jashift,ka+kashift,la+lashift);
		d4f[i][j][k][l] += v;
	}
}
void Derivatives::addDerivesijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor)
{
	if(df && deriv9.df)
		addDijk(deriv9, indexI, indexJ, indexK, scalingFactor);
	if(d2f && deriv9.d2f)
		addD2ijk(deriv9, indexI, indexJ, indexK, scalingFactor);
	if(d3f && deriv9.d3f)
		addD3ijk(deriv9, indexI, indexJ, indexK, scalingFactor);
	if(d4f && deriv9.d4f)
		addD4ijk(deriv9, indexI, indexJ, indexK, scalingFactor);
}
void Derivatives::addDerives(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor)
{
	addDerivesijk( deriv9, indexI, indexJ, indexK, scalingFactor);
}
void Derivatives::addDij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=3;iashift+=3)
	for(int ia=0;ia<3;ia++)
	{
		int i = getIndex(indexI, indexJ, iashift) + ia;
		v = scalingFactor*deriv9.df[ia+iashift];
		df[i] += v;
	}
}

void Derivatives::addD2ij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=3;iashift+=3)
	for(int jashift=0;jashift<=3;jashift+=3)
	for(int ia=0;ia<3;ia++)
	for(int ja=0;ja<3;ja++)
	{
		int i = getIndex(indexI, indexJ, iashift) + ia;
		int j = getIndex(indexI, indexJ, jashift) + ja;
		if(j>i) continue;
		v = scalingFactor*deriv9(ia+iashift,ja+jashift);
		d2f[i][j] += v;
	}
}
void Derivatives::addD3ij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=3;iashift+=3)
	for(int jashift=0;jashift<=3;jashift+=3)
	for(int kashift=0;kashift<=3;kashift+=3)
	for(int ia=0;ia<3;ia++)
	for(int ja=0;ja<3;ja++)
	for(int ka=0;ka<3;ka++)
	{
		int i = getIndex(indexI, indexJ, iashift) + ia;
		int j = getIndex(indexI, indexJ, jashift) + ja;
		if(j>i) continue;
		int k = getIndex(indexI, indexJ, kashift) + ka;
		if(k>j) continue;
		v = scalingFactor*deriv9(ia+iashift,ja+jashift,ka+kashift);
		d3f[i][j][k] += v;
	}
}
void Derivatives::addD4ij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor)
{
	double v;
	for(int iashift=0;iashift<=3;iashift+=3)
	for(int jashift=0;jashift<=3;jashift+=3)
	for(int kashift=0;kashift<=3;kashift+=3)
	for(int lashift=0;lashift<=3;lashift+=3)
	for(int ia=0;ia<3;ia++)
	for(int ja=0;ja<3;ja++)
	for(int ka=0;ka<3;ka++)
	for(int la=0;la<3;la++)
	{
		int i = getIndex(indexI, indexJ, iashift) + ia;
		int j = getIndex(indexI, indexJ, jashift) + ja;
		if(j>i) continue;
		int k = getIndex(indexI, indexJ, kashift) + ka;
		if(k>j) continue;
		int l = getIndex(indexI, indexJ, lashift) + la;
		if(l>k) continue;
		v = scalingFactor*deriv9(ia+iashift,ja+jashift,ka+kashift,la+lashift);
		d4f[i][j][k][l] += v;
	}
}
void Derivatives::addDerivesij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor)
{
	if(df && deriv9.df)
		addDij(deriv9, indexI, indexJ, scalingFactor);
	if(d2f && deriv9.d2f)
		addD2ij(deriv9, indexI, indexJ, scalingFactor);
	if(d3f && deriv9.d3f)
		addD3ij(deriv9, indexI, indexJ, scalingFactor);
	if(d4f && deriv9.d4f)
		addD4ij(deriv9, indexI, indexJ, scalingFactor);
}
void Derivatives::addDerives(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor)
{
	addDerivesij( deriv9, indexI, indexJ, scalingFactor);
}
void Derivatives::add(double v)
{
	value += v;
}
void Derivatives::add(int i, double v)
{
	df[i] += v;
}

void Derivatives::add(int i, int j, double v)
{
/*
	if(i==j) v/=2;// 2!
	d2f[i][j] += v;
*/
	d2f[j][i] += v;
}
void Derivatives::add(int i, int j, int k, double v)
{
	d3f[i][j][k] +=v;
}
void Derivatives::add(int i, int j, int k, int l, double v)
{
	d4f[i][j][k][l] +=v;
}
static int getP(int k, int l=-1, int m=-1, int n=-1)
{
	int N=4;
	if(n<0) N = 3;
	if(m<0) N = 2;
	if(l<0) N = 1;
	if(k<0) return 1;
	int arr[] = {k,l,m, n};
	// N = 2 : npairs = 0 or 1
	// N = 3 : npairs = 0 or 1 or 3
	// N = 4 : npairs = 0 or 1 or 2 or 3 or 6
	int npairs = 0;
	for(int s1=0;s1<N;s1++) for(int s2=s1+1;s2<N;s2++) if(arr[s1]==arr[s2]) npairs++;


	if(npairs==6) return 24; // 4!
	else if(npairs==3) return 6; // 3!
	else if(npairs==2) return 4; // 2!2!
	else if(npairs==1) return 2; // 2!
	else if(npairs !=0) { cerr<<"Interior Error"<<endl; exit(1);}
	return 1;
}
/* 
	F(G1,G2,...GN)
	Gi(x1,x2,.... xM)
	add contribution fo dnF/dxn using dnF/dGn & dnG/dxn
*/
void Derivatives::addTod1F(const Derivatives& dnFGn, const Derivatives* G, int* index)
{
	int k = index[0];

	if(!dnFGn.df) return;

	if(df && G[0].df)
	for(int a=0;a<nVariables;a++) 
		df[a] += dnFGn.df[k]*G[0].df[a];

	if(d2f && G[0].d2f)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
		{
			double v = dnFGn.df[k]*G[0].d2f[a][b];
			d2f[a][b] += v;
		}

	if(d3f && G[0].d3f)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
			for(int c=0;c<=b;c++) 
			{
				double v = dnFGn.df[k]*G[0].d3f[a][b][c];
                                d3f[a][b][c] += v;
			}
	if(d4f && G[0].d4f)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
			for(int c=0;c<=b;c++) 
			for(int d=0;d<=c;d++) 
			{
				double v = dnFGn.df[k]*G[0].d4f[a][b][c][d];
                                d4f[a][b][c][d] += v;
			}
}
void Derivatives::addTod2F(const Derivatives& dnFGn, const Derivatives* G, int* index)
{
	int k = index[0];
	int l = index[1];
	if(!dnFGn.d2f) return;

	if(d2f && G[0].df && G[1].df)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
		{
			double v = dnFGn.d2f[k][l]*G[0].df[a]*G[1].df[b];
			       v+= dnFGn.d2f[k][l]*G[1].df[a]*G[0].df[b];
			if(k==l) v/=2; //2!
			d2f[a][b] += v;
		}

	if(d3f && G[0].df && G[1].df && G[0].d2f && G[1].d2f)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
			for(int c=0;c<=b;c++) 
			{
				double v = dnFGn.d2f[k][l]*(
						G[0].d2f[b][c]*G[1].df[a]+
						G[0].d2f[a][c]*G[1].df[b]+
						G[0].d2f[a][b]*G[1].df[c]
					   );
					v += dnFGn.d2f[k][l]*(
						G[1].d2f[b][c]*G[0].df[a]+
						G[1].d2f[a][c]*G[0].df[b]+
						G[1].d2f[a][b]*G[0].df[c]
					   );
				if(k==l) v/=2; //2!
                                d3f[a][b][c] += v;
			}
	if(d4f && G[0].df && G[1].df && G[0].d2f && G[1].d2f && G[0].d3f && G[1].d3f)
	{
		for(int a=0;a<nVariables;a++) 
			for(int b=0;b<=a;b++) 
				for(int c=0;c<=b;c++) 
					for(int d=0;d<=c;d++) 
				{
					double v = 0;
					v += dnFGn.d2f[k][l]*(
						G[0].d2f[a][b]*G[1].d2f[c][d]+
						G[0].d2f[a][c]*G[1].d2f[b][d]+
						G[0].d2f[a][d]*G[1].d2f[b][c]+

						G[0].d3f[a][b][c]*G[1].df[d]+
						G[0].d3f[a][b][d]*G[1].df[c]+
						G[0].d3f[a][c][d]*G[1].df[b]+
						G[0].d3f[b][c][d]*G[1].df[a]
					   	);
					v += dnFGn.d2f[k][l]*(
						G[1].d2f[a][b]*G[0].d2f[c][d]+
						G[1].d2f[a][c]*G[0].d2f[b][d]+
						G[1].d2f[a][d]*G[0].d2f[b][c]+

						G[1].d3f[a][b][c]*G[0].df[d]+
						G[1].d3f[a][b][d]*G[0].df[c]+
						G[1].d3f[a][c][d]*G[0].df[b]+
						G[1].d3f[b][c][d]*G[0].df[a]
					   	);
					if(k==l) v/=2; //2!
                                	d4f[a][b][c][d] += v;
				}
	}
}
void Derivatives::addTod3F(const Derivatives& dnFGn, const Derivatives* G, int* index)
{
	int k = index[0];
	int l = index[1];
	int m = index[2];

	if(!dnFGn.d3f) return;
	if(!d3f) return;

	int N = 3;
	int Np;
	int A[] = {k,l,m};
	int B[] = {0,1,2};
	int** listPerm3B = getListPermutation(B, N, Np);
	double f3 = 1.0/getP(k, l,m);

	if(d3f && G[0].df  && G[1].df && G[2].df)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
			for(int c=0;c<=b;c++) 
			{
				double v = 0;
				for(int i=0;i<Np;i++)
				{
					int kB = listPerm3B[i][0];
					int lB = listPerm3B[i][1];
					int mB = listPerm3B[i][2];

					v += dnFGn.d3f[k][l][m]*(G[kB].df[a]*G[lB].df[b]*G[mB].df[c]);
				}
				v *= f3;
                                d3f[a][b][c] += v;
			}
	if(d4f && G[0].df  && G[1].df && G[2].df && G[0].d2f  && G[1].d2f && G[2].d2f)
	for(int a=0;a<nVariables;a++) 
		for(int b=0;b<=a;b++) 
			for(int c=0;c<=b;c++) 
				for(int d=0;d<=c;d++) 
			{
				double v = 0;
				for(int i=0;i<Np;i++)
				{
					int kB = listPerm3B[i][0];
					int lB = listPerm3B[i][1];
					int mB = listPerm3B[i][2];
					v += dnFGn.d3f[k][l][m]*(
						G[kB].d2f[a][b]*G[lB].df[c]*G[mB].df[d]+
						G[kB].d2f[a][c]*G[lB].df[b]*G[mB].df[d]+
						G[kB].d2f[a][d]*G[lB].df[b]*G[mB].df[c]+
						G[kB].d2f[b][c]*G[lB].df[a]*G[mB].df[d]+
						G[kB].d2f[b][d]*G[lB].df[a]*G[mB].df[c]+
						G[kB].d2f[c][d]*G[lB].df[a]*G[mB].df[b]
					   );
				}
				v *= f3;
                                d4f[a][b][c][d] += v;
			}
	free2Dtable(Np, listPerm3B);
}
void Derivatives::addTod4F(const Derivatives& dnFGn, const Derivatives* G, int* index)
{
	int k = index[0];
	int l = index[1];
	int m = index[2];
	int n = index[3];

	if(!dnFGn.d4f) return;
	if(!d4f) return;

	int N = 4;
	int Np;
	int A[] = {k,l,m,n};
	int B[] = {0,1,2,3};
	int** listPerm4B = getListPermutation(B, N, Np);
	double f4 = 1.0/getP(k, l,m, n);

	if(d4f && G[0].df  && G[1].df && G[2].df && G[3].df)
		for(int a=0;a<nVariables;a++) 
			for(int b=0;b<=a;b++) 
				for(int c=0;c<=b;c++) 
					for(int d=0;d<=c;d++) 
			{
				double v = 0;
				for(int i=0;i<Np;i++)
				{
					int kB = listPerm4B[i][0];
					int lB = listPerm4B[i][1];
					int mB = listPerm4B[i][2];
					int nB = listPerm4B[i][3];

					v += dnFGn.d4f[k][l][m][n]*(
						G[kB].df[a]*G[lB].df[b]*G[mB].df[c]*G[nB].df[d]
					   );
				}
				v *= f4;
                                d4f[a][b][c][d] += v;
			}
	free2Dtable(Np, listPerm4B);
}
void Derivatives::addTodF(const Derivatives& dnFGn, const Derivatives* G, int* index, int order=-1)
{
	if(nVariables != G[0].nVariables || maxOrder != G[0].maxOrder ) 
	{
		free();
		*this = Derivatives(G[0].maxOrder, G[0].nVariables);
		reset();
	}
	if(order==1 || order <0)
		 if(df) addTod1F(dnFGn, G, index);
	if(order==2 || order <0)
		if(d2f) addTod2F(dnFGn, G, index);
	if(order==3 || order <0)
		if(d3f) addTod3F(dnFGn, G, index);
	if(order==4 || order <0)
		if(d4f) addTod4F(dnFGn, G, index);
	value = dnFGn.value;
}
void Derivatives::compute(const NeuralNetwork* neuralNetwork)
{
	int nG = neuralNetwork->getNumNeurons(0);
	if(nVariables < nG) 
	{
		int mOrder = maxOrder;
		free();
		*this = Derivatives(mOrder, nG);
		reset();
	}
	neuralNetwork->calculateDnEdGn(df,d2f,d3f,d4f);
	neuralNetwork->getOutput(&value);
}
// convProp : convEnergy or convDipole
void Derivatives::toPhysicalUnits(double convProp, double convLength)
{
	double s0 = 1.0/convProp;
	double s1 = s0*convLength;
	double s2 = s1*convLength;
	double s3 = s2*convLength;
	double s4 = s3*convLength;
	
	value *= s0; 
	if(df)
	for(int a=0;a<nVariables;a++) df[a] *= s1;

	if(d2f)
	for(int a=0;a<nVariables;a++) 
	for(int b=0;b<=a;b++) d2f[a][b] *= s2;

	if(d3f)
	for(int a=0;a<nVariables;a++) 
	for(int b=0;b<=a;b++) 
	for(int c=0;c<=b;c++) d3f[a][b][c] *= s3;

	if(d4f)
	for(int a=0;a<nVariables;a++) 
	for(int b=0;b<=a;b++) 
	for(int c=0;c<=b;c++) 
	for(int d=0;d<=c;d++) d4f[a][b][c][d] *= s4;
}

static void swap(int *a, int *b)
{
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
static void getListPermutation(int *arr, int start, int end, int** list, int& il)
{
    if(start==end)
    {
        //cout<<"il="<<il<<endl;
        for(int i=0;i<=end;i++) list[il][i] = arr[i];
        il++;
        return;
    }
    for(int i=start;i<=end;i++)
    {
        //swapping numbers
        swap((arr+i), (arr+start));
        //fixing one first digit
        //and calling getListPermutation on
        //the rest of the digits
        getListPermutation(arr, start+1, end, list, il);
        swap((arr+i), (arr+start));
    }
}
static int factorial(int n)
{
        int f = 1;
        for(int i=2;i<=n;i++) f *= i;
        return f;
}
static int **getListPermutation(int *arr, int n, int& np)
{
        np = factorial(n);// n!
        int **list = new int*[np];
        for( int i=0;i<np;i++)
                list[i] = new int [n];
        int il = 0;
        getListPermutation(arr, 0, n-1, list, il);
        return list;
}
static double* new1Dtable(int size)
{
	double* t = new double[size];
	return t;
}
static double* copy1Dtable(int size, double *tc)
{
	double* t = new1Dtable(size);
    	for (int i = 0; i < size; i++) t[i]  = tc[i];
	return t;
}
static void copy1Dtable(int size, double *tc, double* t)
{
    	for (int i = 0; i < size; i++) t[i]  = tc[i];
}
static void free1Dtable(double* t)
{
	delete[] t;
}
static double** new2Dtable(int size)
{
    double** t = new double*[size];
    for (int l = 0; l < size; l++) t[l]  = new double[l+1];
    return t;
}
static double** copy2Dtable(int size, double **tc)
{
	double** t = new2Dtable(size);
    	for (int i = 0; i < size; i++) 
    	for (int j = 0; j <=i; j++) 
		t[i][j]  = tc[i][j];
	return t;
}
static void copy2Dtable(int size, double **tc, double** t)
{
    	for (int i = 0; i < size; i++) 
    	for (int j = 0; j <=i; j++) 
		t[i][j]  = tc[i][j];
}
static void free2Dtable(int size, double** t)
{
    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
static void free2Dtable(int size, int** t)// int table
{
    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
static double*** new3Dtable(int size)
{
    double*** t = new double**[size];
    for (int l = 0; l < size; l++) 
		t[l] = new double*[l+1];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m <=l; m++) 
		t[l][m] = new double[m+1];
    return t;
}
static double*** copy3Dtable(int size, double ***tc)
{
	double*** t = new3Dtable(size);
    	for (int i = 0; i < size; i++) 
    	for (int j = 0; j <=i; j++) 
    	for (int k = 0; k <=j; k++) 
		t[i][j][k]  = tc[i][j][k];
	return t;
}
static void copy3Dtable(int size, double*** tc, double*** t)
{
    	for (int i = 0; i < size; i++) 
    	for (int j = 0; j <=i; j++) 
    	for (int k = 0; k <=j; k++) 
		t[i][j][k]  = tc[i][j][k];
}
static void free3Dtable(int size, double*** t)
{
    for (int l = 0; l < size; l++) 
    	for (int m = 0; m <=l; m++) 
		delete [] t[l][m];
    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
static double**** new4Dtable(int size)
{
    double**** t = new double***[size];
    for (int l = 0; l < size; l++) 
		t[l] = new double**[l+1];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m <=l; m++) 
		t[l][m] = new double*[m+1];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m <=l; m++) 
    		for (int n = 0; n <=m; n++) 
		t[l][m][n] = new double[n+1];
    return t;
}
static double**** copy4Dtable(int size, double ****tc)
{
	double**** t = new4Dtable(size);
    	for (int i = 0; i < size; i++) 
    	for (int j = 0; j <=i; j++) 
    	for (int k = 0; k <=j; k++) 
    	for (int l = 0; l <=k; l++) 
		t[i][j][k][l]  = tc[i][j][k][l];
	return t;
}
static void copy4Dtable(int size, double**** tc, double**** t)
{
    	for (int i = 0; i < size; i++) 
    	for (int j = 0; j <=i; j++) 
    	for (int k = 0; k <=j; k++) 
    	for (int l = 0; l <=k; l++) 
		t[i][j][k][l]  = tc[i][j][k][l];
}
static void free4Dtable(int size, double**** t)
{
    for (int l = 0; l < size; l++) 
    	for (int m = 0; m <=l; m++) 
    		for (int n = 0; n <=m; n++) 
		delete [] t[l][m][n];

    for (int l = 0; l < size; l++) 
    	for (int m = 0; m <=l; m++) 
		delete [] t[l][m];

    for (int l = 0; l < size; l++) delete[] t[l];
    delete[] t;
}
static void build9From6ij(double* d6, double* d9)
{
	for(size_t l=0;l<6;l++) d9[l] = d6[l];
	for(size_t l=6;l<9;l++) d9[l] = 0;
}
static void build9From6ik(double* d6, double* d9)
{
	for(size_t l=0;l<3;l++) d9[l] = d6[l];
	for(size_t l=3;l<6;l++) d9[l] = 0;
	for(size_t l=6;l<9;l++) d9[l] = d6[l-3];
}
static void build9From6jk(double* d6, double* d9)
{
	for(size_t l=0;l<3;l++) d9[l] = 0;
	for(size_t l=3;l<9;l++) d9[l] = d6[l-3];
}
static void build9From6ij(double** d6, double** d9)
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<=l;m++) d9[l][m] = 0;
	for(size_t l=0;l<6;l++) for(size_t m=0;m<=l;m++) d9[l][m] = d6[l][m];
}
static void build9From6ik(double** d6, double** d9)
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<=l;m++) d9[l][m] = 0;
	for(size_t l=0;l<3;l++) for(size_t m=0;m<=l;m++) d9[l][m] = d6[l][m];
	for(size_t l=0;l<3;l++) for(size_t m=6;m<9;m++) d9[m][l] = d6[m-3][l];
	for(size_t l=6;l<9;l++) for(size_t m=6;m<=l;m++) d9[l][m] = d6[l-3][m-3];
}
static void build9From6jk(double** d6, double** d9)
{
	for(size_t l=0;l<9;l++) for(size_t m=0;m<=l;m++) d9[l][m] = 0;
	for(size_t l=3;l<9;l++) for(size_t m=3;m<=l;m++) d9[l][m] = d6[l-3][m-3];
}
static void build9From6ij(double*** d6, double*** d9)
{
	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<=l;m++) 
			for(size_t n=0;n<=m;n++) 
				d9[l][m][n] = 0;
	for(size_t l=0;l<6;l++) 
		for(size_t m=0;m<=l;m++) 
			for(size_t n=0;n<=m;n++) 
			d9[l][m][n] = d6[l][m][n];
}
static void build9From6ij(double**** d6, double**** d9)
{
	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<=l;m++) 
			for(size_t n=0;n<=m;n++) 
				for(size_t k=0;k<=n;k++) 
					d9[l][m][n][k] = 0;
	for(size_t l=0;l<6;l++) 
		for(size_t m=0;m<=l;m++) 
			for(size_t n=0;n<=m;n++) 
				for(size_t k=0;k<=n;k++) 
					d9[l][m][n][k] =  d6[l][m][n][k];
}
static void build9From6ik(double*** d6, double*** d9)
{
	for(int l=0;l<9;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				d9[l][m][n] = 0;
	for(int lmin=0;lmin<=3;lmin+=3) 
	for(int l=lmin;l<lmin+3;l++) 
		for(int mmin=0;mmin<=3;mmin+=3) 
		for(int m=mmin;m<mmin+3;m++) 
			for(int nmin=0;nmin<=3;nmin+=3) 
			for(int n=nmin;n<nmin+3;n++) 
				{

					if(n>m) continue;
					if(m>l) continue;
					int i =l-lmin;
					int j =m-mmin;
					int k =n-nmin;
					if(k>j) { int t = k; k=j; j = t; }
					if(j>i) { int t = i; i=j; j = t; }
					if(k>j) { int t = k; k=j; j = t; }

					double v =d6[i][j][k];
					d9[l][m][n] = v;
				}
}
static void build9From6ik(double**** d6, double**** d9)
{
	for(int l=0;l<9;l++) 
		for(int m=0;m<=l;m++) 
			for(int n=0;n<=m;n++) 
				for(int k=0;k<=n;k++) 
					d9[l][m][n][k] = 0;
	for(int lmin=0;lmin<=3;lmin+=3) 
	for(int l=lmin;l<lmin+3;l++) 
		for(int mmin=0;mmin<=3;mmin+=3) 
		for(int m=mmin;m<mmin+3;m++) 
			for(int nmin=0;nmin<=3;nmin+=3) 
			for(int n=nmin;n<nmin+3;n++) 
				for(int kmin=0;kmin<=3;kmin+=3) 
				for(int k=kmin;k<kmin+3;k++) 
				{

					if(k>n) continue;
					if(n>m) continue;
					if(m>l) continue;
					int a =l-lmin;
					int b =m-mmin;
					int c =n-nmin;
					int d =k-kmin;
					if(d>c) { int t = c; c=d; d = t; }
					if(c>b) { int t = c; c=b; b = t; }
					if(b>a) { int t = a; a=b; b = t; }
					if(d>c) { int t = c; c=d; d = t; }
					if(c>b) { int t = c; c=b; b = t; }
					if(d>c) { int t = c; c=d; d = t; }
					d9[l][m][n][k] = d6[a][b][c][d];
				}
}
static void build9From6jk(double*** d6, double*** d9)
{
	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<=l;m++) 
			for(size_t n=0;n<=m;n++) 
				d9[l][m][n] = 0;
	for(size_t l=3;l<9;l++) 
		for(size_t m=3;m<=l;m++) 
			for(size_t n=3;n<=m;n++) 
			d9[l][m][n] = d6[l-3][m-3][n-3];
}
static void build9From6jk(double**** d6, double**** d9)
{
	for(size_t l=0;l<9;l++) 
		for(size_t m=0;m<=l;m++) 
			for(size_t n=0;n<=m;n++) 
				for(size_t k=0;k<=n;k++) 
					d9[l][m][n][k] = 0;
	for(size_t l=3;l<9;l++) 
		for(size_t m=3;m<=l;m++) 
			for(size_t n=3;n<=m;n++) 
				for(size_t k=3;k<=n;k++) 
					d9[l][m][n][k] =  d6[l-3][m-3][n-3][k-3];
}
static void build9From6(int lm,  double* d6, double* d9)
{
	if(lm==0) build9From6ij(d6, d9);
	else if(lm==1) build9From6ik(d6, d9);
	else if(lm==2) build9From6jk(d6, d9);
}
static void build9From6(int lm, double** d6, double** d9)
{
	if(lm==0) build9From6ij(d6, d9);
	else if(lm==1) build9From6ik(d6, d9);
	else if(lm==2) build9From6jk(d6, d9);
}
static void build9From6(int lm, double*** d6, double*** d9)
{
	if(lm==0) build9From6ij(d6, d9);
	else if(lm==1) build9From6ik(d6, d9);
	else if(lm==2) build9From6jk(d6, d9);
}
static void build9From6(int lm, double**** d6, double**** d9)
{
	if(lm==0) build9From6ij(d6, d9);
	else if(lm==1) build9From6ik(d6, d9);
	else if(lm==2) build9From6jk(d6, d9);

}
bool Derivatives::save(const char* fileName)
{
	ofstream file(fileName, ios::out| ios::binary);
	if(file.fail()) return false;
	file.write((char*)&maxOrder, sizeof(int));
	file.write((char*)&nVariables, sizeof(int));
	file.write((char*)&value, sizeof(double));
	if(df) file.write((char*)df, nVariables*sizeof(double));
	if(d2f) 
	{
		for(int i=0;i<nVariables;i++)
			file.write((char*)d2f[i], (i+1)*sizeof(double));
	}
	if(d3f) 
	{
		for(int i=0;i<nVariables;i++)
		for(int j=0;j<=i;j++)
			file.write((char*)d3f[i][j], (j+1)*sizeof(double));
	}
	if(d4f) 
	{
		for(int i=0;i<nVariables;i++)
		for(int j=0;j<=i;j++)
		for(int k=0;k<=j;k++)
			file.write((char*)d4f[i][j][k], (k+1)*sizeof(double));
	}
	file.close();
	return true;
}
bool Derivatives::read(const char* fileName)
{
	ifstream file(fileName, ios::in| ios::binary);
	if(file.fail()) return false;
	int mOrder;
	file.read((char*)&mOrder, sizeof(int));
	int nV;
	file.read((char*)&nV, sizeof(int));

	if(mOrder != maxOrder || nV != nVariables)
	{
		free();
		maxOrder = mOrder;
		nVariables = nV;
		if(nVariables<1) return true;
		if(maxOrder>0) df = new1Dtable(nVariables);
		if(maxOrder>1) d2f = new2Dtable(nVariables);
		if(maxOrder>2) d3f = new3Dtable(nVariables);
		if(maxOrder>3) d4f = new4Dtable(nVariables);
	}

	file.read((char*)&value, sizeof(double));

	if(df) file.read((char*)df, nVariables*sizeof(double));
	if(d2f) 
	{
		for(int i=0;i<nVariables;i++)
			file.read((char*)d2f[i], (i+1)*sizeof(double));
	}
	if(d3f) 
	{
		for(int i=0;i<nVariables;i++)
		for(int j=0;j<=i;j++)
			file.read((char*)d3f[i][j], (j+1)*sizeof(double));
	}
	if(d4f) 
	{
		for(int i=0;i<nVariables;i++)
		for(int j=0;j<=i;j++)
		for(int k=0;k<=j;k++)
			file.read((char*)d4f[i][j][k], (k+1)*sizeof(double));
	}
	file.close();
	return true;
}

#endif /* HIGH_DERIVATIVES*/
