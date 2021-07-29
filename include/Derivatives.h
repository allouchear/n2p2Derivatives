// Abdul-Rahman Allouche
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

#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#ifdef HIGH_DERIVATIVES

#include <cstdlib>  
#include <stdexcept>      // std::invalid_argument
#include "CutoffFunction.h"
#include "NeuralNetwork.h"
#include <iostream>  


namespace nnp
{

class Derivatives
{
public:
       /** MaxOrder : 1, 2 , 3, or 4
       */
	int maxOrder;
	int nVariables;
	/** value of the function
	*/
	double value;
	/** first derivatives
	*/
	double* df;
	// only lower part is created and used, assuming a symetric matrix
	double** d2f;
	double*** d3f;
	double**** d4f;

    	Derivatives();
    	Derivatives(int mOrder, int nVar);
    	Derivatives(int mOrder);
    	Derivatives(const Derivatives& right);
    	~Derivatives();
	Derivatives& operator=(const Derivatives& right);
	int getMaxOrder() const;
	void reset();


	void computeSum(double a, double b, const Derivatives & f2, Derivatives & fsum) const;
	void computeSum(const Derivatives & f2, Derivatives & fsum) const;
	Derivatives computeSum(double a, double b, const Derivatives & f2) const;
	Derivatives computeSum(const Derivatives & f2) const;
	Derivatives operator+(const Derivatives & f2) const;
	void operator+=(const Derivatives & f2);
	void operator+=(double v);
	void computeProd(const Derivatives& f2, Derivatives& fprod) const;
	Derivatives computeProd(const Derivatives& f2) const;
	Derivatives operator*(const Derivatives& f2) const;
	void operator*=(const Derivatives& f2);
	void computeProd(double s, Derivatives& fprod) const;
	Derivatives computeProd(double s) const;
	Derivatives operator*(double s) const;
	void operator*=(double s);
	void computeProd(const Derivatives& f2, const Derivatives& f3, Derivatives& fprod);
	Derivatives computeProd(const Derivatives& f2, const Derivatives& f3);
	void computednFzeta(const Derivatives& F, double zeta);
	Derivatives getdnFzeta(double zeta) const;
	void computednfu(const double* dnf, const Derivatives& u);
	void computednRijk(const double* rij, const double* rik);
    	Derivatives(int mOrder, const double* rij, const double* rik);
	void computednRij2_6(const double* rij);
	void computednR2_9(int lm, const double* r);
	void computednR2(int lm, const double* r);
    	Derivatives(int mOrder, int lm, const double* r);
	void computecosijk(const double* rij, const double* rik);
	void computecosijk(double zeta, double lambda, const double* rij, const double* rik);
	void computednfc(const CutoffFunction& cf, int lm, const double* r);
	void computednfc(const CutoffFunction& cf, const double* rij, const double* rik);
	void computednfc(const CutoffFunction& cf, const double* rij, const double* rik, const double* rjk);
	void computednexpu(const Derivatives& u);
	void computednRij2pRik2(double aij, double aik, const double* rij, const double* rik);
	void computednexpmetaRij2pRik2(double eta, const double* rij, const double* rik);
	void computednRij2pRik2pRjk2(double aij, double aik, double ajk, const double* rij, const double* rik, const double* rjk);
	void computednexpmetaRij2pRik2pRjk2(double eta, const double* rij, const double* rik, const double* rjk);
	void computednexpmetaR2(int lm, double eta, const double* r);
	Derivatives(int mOrder, int lm, const double* r, double rs);
	void computednRij2pRik2(double aij, double aik, const double* rij, const double* rik, double rs);
	void computednexpmetaRij2pRik2(double eta, const double* rij, const double* rik, double rs);
	void computednRij2pRik2pRjk2(double aij, double aik, double ajk, const double* rij, const double* rik, const double* rjk, double rs);
	void computednexpmetaRij2pRik2pRjk2(double eta, const double* rij, const double* rik, const double* rjk, double rs);
	void computednexpmetaR2(int lm, double eta, const double* r, double rs);
	void computeG(int lm, double eta, const double* r, double rs, const CutoffFunction& fc);
	void computeG(double eta, const double* rij, double rs, const CutoffFunction& fc);
	Derivatives(int maxOrder, int lm, double eta, const double* r, double rs, const CutoffFunction& fc);
	Derivatives(int maxOrder, double eta, const double* rij, double rs, const CutoffFunction& fc);
	void computeG(double zeta, double lambda, double eta, const double* rij, const double* rik, const double* rjk, double rs, const CutoffFunction& cf);
	Derivatives(int maxOrder, double zeta, double lambda, double eta, const double* rij, const double* rik, const double* rjk, double rs, const CutoffFunction& cf);
	void computeG(double zeta, double lambda, double eta, const double* rij, const double* rik, double rs, const CutoffFunction& cf);
	Derivatives(int maxOrder, double zeta, double lambda, double eta, const double* rij, const double* rik, double rs, const CutoffFunction& cf);

	void addDerivesijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor);
	void addDerives(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor);
	void addDerivesij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor);
	void addDerives(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor);

	void add(double v);
	void add(int i, double v);
	void add(int i, int j, double v);
	void add(int i, int j, int k, double v);
	void add(int i, int j, int k, int l, double v);

	void addTodF(const Derivatives& dnFGn, const Derivatives* G, int* index, int order);

	double operator()() const;
	double operator()(int i) const;
	double operator()(int i, int j) const;
	double operator()(int i, int j, int k) const;
	double operator()(int i, int j, int k, int l) const;
	void compute(const NeuralNetwork* neuralNetwork);
	void toPhysicalUnits(double convProp, double convLength);
	bool save(const char* fileName);
	bool read(const char* fileName);
	void setValue(double val);
	void setdFValue(double val, int index);
	void setd2FValue(double val, int indexi, int indexj);

private:
	void free();
	void resetdf();
	void resetd2f();
	void resetd3f();
	void resetd4f();
	void computeSumFirst(double a, double b, const Derivatives & f2, Derivatives & fsum) const;
	void computeSumSecond(double a, double b, const Derivatives & f2, Derivatives & fsum) const;
	void computeSumThird(double a, double b, const Derivatives & f2, Derivatives & fsum) const;
	void computeSumFourth(double a, double b, const Derivatives & f2, Derivatives & fsum) const;
	void computeProdFirst(const Derivatives& f2, Derivatives& fprod) const;
	void computeProdSecond(const Derivatives& f2, Derivatives& fprod) const;
	void computeProdThird(const Derivatives& f2, Derivatives& fprod) const;
	void computeProdFourth(const Derivatives& f2, Derivatives& fprod) const;
	void computeProdFirst(double s, Derivatives& fprod) const;
	void computeProdSecond(double s, Derivatives& fprod) const;
	void computeProdThird(double s, Derivatives& fprod) const;
	void computeProdFourth(double s, Derivatives& fprod) const;
	void computedfu(const double* dnf, const Derivatives& u);
	void computed2fu(const double* dnf, const Derivatives& u);
	void computed3fu(const double* dnf, const Derivatives& u);
	void computed4fu(const double* dnf, const Derivatives& u);
	void computedFzeta(const Derivatives& F, double zeta);
	void computed2Fzeta(const Derivatives& F, double zeta);
	void computed3Fzeta(const Derivatives& F, double zeta);
	void computed4Fzeta(const Derivatives& F, double zeta);
	void computedRijk(const double* rij, const double* rik);
	void computed2Rijk();
	void computed3Rijk();
	void computed4Rijk();
	void computedRij2_6(const double* rij);
	void computed2Rij2_6();
	void computed3Rij2_6();
	void computed4Rij2_6();
	void build9VFrom6V(int lm, Derivatives& d6);
	void addDijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor);
	void addD2ijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor);
	void addD3ijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor);
	void addD4ijk(const Derivatives& deriv9, int indexI, int indexJ, int indexK, double scalingFactor);
	void addDij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor);
	void addD2ij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor);
	void addD3ij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor);
	void addD4ij(const Derivatives& deriv9, int indexI, int indexJ, double scalingFactor);
	void addTod1F(const Derivatives& dnFGn, const Derivatives* G, int* index);
	void addTod2F(const Derivatives& dnFGn, const Derivatives* G, int* index);
	void addTod3F(const Derivatives& dnFGn, const Derivatives* G, int* index);
	void addTod4F(const Derivatives& dnFGn, const Derivatives* G, int* index);

};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////
inline int Derivatives::getMaxOrder() const
{
	return maxOrder;
}
inline 
double Derivatives::operator()() const
{
	return value;
}
inline 
double Derivatives::operator()(int i) const
{
	return df[i];
}
inline 
double Derivatives::operator()(int i, int j) const
{
	if(i>j) return d2f[i][j];
	else return d2f[j][i];
}
inline 
double Derivatives::operator()(int i, int j, int k) const
{
	if(k>j) { int t = k; k=j; j = t; }
	if(j>i) { int t = i; i=j; j = t; }
	if(k>j) { int t = k; k=j; j = t; }
	return d3f[i][j][k];
}
inline 
double Derivatives::operator()(int i, int j, int k, int l) const
{
	if(l>k) { int t = k; k=l; l = t; }
	if(k>j) { int t = k; k=j; j = t; }
	if(j>i) { int t = i; i=j; j = t; }
	if(l>k) { int t = k; k=l; l = t; }
	if(k>j) { int t = k; k=j; j = t; }
	if(l>k) { int t = k; k=l; l = t; }
	/*
	if(j>i || k>j || l>k) 
	{
		std::cerr<<"Internal error"<<std::endl;
		std::cerr<<"ijkl ="<<i<<" "<<j<<" "<<k<<" "<<l<<std::endl;
		exit(1);
	}
	*/
	return d4f[i][j][k][l];
}
inline 
void Derivatives::setValue(double val)
{
	value = val;
}
inline 
void Derivatives::setdFValue(double val, int index)
{
	if(df && index>=0 && index<nVariables) df[index] = val;
}
inline 
void Derivatives::setd2FValue(double val, int indexi, int indexj)
{
	if(d2f && indexi>=0 && indexi<nVariables && indexj>=0 && indexj<nVariables ) d2f[indexi][indexj] = d2f[indexj][indexi]  = val;
}


#endif /* HIGH_DERIVATIVES */

}
#endif
