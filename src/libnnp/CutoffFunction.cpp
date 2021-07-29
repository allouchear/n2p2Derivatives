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

#include "CutoffFunction.h"
#include "utilityDerivatives.h"
#include <stdexcept>
#include <cmath>  // cos, sin, tanh, exp, pow
#include <limits> // std::numeric_limits

using namespace std;
using namespace nnp;

double const CutoffFunction::PI = 4.0 * atan(1.0);
double const CutoffFunction::PI_2 = 2.0 * atan(1.0);
double const CutoffFunction::E = exp(1.0);
double const CutoffFunction::TANH_PRE = pow((E + 1 / E) / (E - 1 / E), 3);

CutoffFunction::CutoffFunction() : cutoffType(CT_HARD                 ),
                                   rc        (0.0                     ),
                                   rcinv     (0.0                     ),
                                   rci       (0.0                     ),
                                   alpha     (0.0                     ),
                                   iw        (0.0                     ),
                                   fPtr      (&CutoffFunction::  fHARD),
                                   dfPtr     (&CutoffFunction:: dfHARD),
                                   fdfPtr    (&CutoffFunction::fdfHARD)
#ifdef HIGH_DERIVATIVES
			  	   ,
                                   dnfPtr     (&CutoffFunction:: dnfHARD)
#endif
{
}

void CutoffFunction::setCutoffType(CutoffType const cutoffType)
{
    this->cutoffType = cutoffType;

    if (cutoffType == CT_HARD)
    {
          fPtr = &CutoffFunction::  fHARD;
         dfPtr = &CutoffFunction:: dfHARD;
        fdfPtr = &CutoffFunction::fdfHARD;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfHARD;
#endif
    }
    else if (cutoffType == CT_COS)
    {
          fPtr = &CutoffFunction::  fCOS;
         dfPtr = &CutoffFunction:: dfCOS;
        fdfPtr = &CutoffFunction::fdfCOS;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfCOS;
#endif
    }
    else if (cutoffType == CT_TANHU)
    {
          fPtr = &CutoffFunction::  fTANHU;
         dfPtr = &CutoffFunction:: dfTANHU;
        fdfPtr = &CutoffFunction::fdfTANHU;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfTANHU;
#endif
    }
    else if (cutoffType == CT_TANH)
    {
          fPtr = &CutoffFunction::  fTANH;
         dfPtr = &CutoffFunction:: dfTANH;
        fdfPtr = &CutoffFunction::fdfTANH;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfTANH;
#endif
    }
    else if (cutoffType == CT_EXP)
    {
          fPtr = &CutoffFunction::  fEXP;
         dfPtr = &CutoffFunction:: dfEXP;
        fdfPtr = &CutoffFunction::fdfEXP;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfEXP;
#endif
    }
    else if (cutoffType == CT_POLY1)
    {
          fPtr = &CutoffFunction::  fPOLY1;
         dfPtr = &CutoffFunction:: dfPOLY1;
        fdfPtr = &CutoffFunction::fdfPOLY1;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfPOLY1;
#endif
    }
    else if (cutoffType == CT_POLY2)
    {
          fPtr = &CutoffFunction::  fPOLY2;
         dfPtr = &CutoffFunction:: dfPOLY2;
        fdfPtr = &CutoffFunction::fdfPOLY2;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfPOLY2;
#endif
    }
    else if (cutoffType == CT_POLY3)
    {
          fPtr = &CutoffFunction::  fPOLY3;
         dfPtr = &CutoffFunction:: dfPOLY3;
        fdfPtr = &CutoffFunction::fdfPOLY3;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfPOLY3;
#endif
    }
    else if (cutoffType == CT_POLY4)
    {
          fPtr = &CutoffFunction::  fPOLY4;
         dfPtr = &CutoffFunction:: dfPOLY4;
        fdfPtr = &CutoffFunction::fdfPOLY4;
#ifdef HIGH_DERIVATIVES
        dnfPtr = &CutoffFunction::dnfPOLY4;
#endif
    }
    else
    {
        throw invalid_argument("ERROR: Unknown cutoff type.\n");
    }

    return;
}

void CutoffFunction::setCutoffRadius(double const cutoffRadius)
{
    rc = cutoffRadius;
    rcinv = 1.0 / cutoffRadius;
    return;
}

void CutoffFunction::setCutoffParameter(double const alpha)
{
    if (alpha < 0.0 || alpha >= 1.0)
    {
        throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
    }
    this->alpha = alpha;
    rci = rc * alpha;
    iw = 1.0 / (rc - rci);
    return;
}

double CutoffFunction::fCOS(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    return 0.5 * (cos(PI * x) + 1.0);
}

double CutoffFunction::dfCOS(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    return -PI_2 * iw * sin(PI * x);
}

void CutoffFunction::fdfCOS(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    double const temp = cos(PI * x);
    fc = 0.5 * (temp + 1.0);
    dfc = -0.5 * iw * PI * sqrt(1.0 - temp * temp);
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfCOS(double r, int order) const
{
    if (r < rci) 
    {
	if(order==0) return 1.0;
	return 0.0;
    }
    double const x = (r - rci) * iw;
    double p = 0.5;
    if(order==0) return p*(cos(PI * x) + 1.0);
    for(int i=1;i<=order;i++) p *= -iw * PI;
    
    if(order%2==0) return p*cos(PI*x);
    else return p*sin(PI*x);
}
#endif

double CutoffFunction::fTANHU(double r) const
{
    double const temp = tanh(1.0 - r * rcinv);
    return temp * temp * temp;
}

double CutoffFunction::dfTANHU(double r) const
{
    double temp = tanh(1.0 - r * rcinv);
    temp *= temp;
    return 3.0 * temp * (temp - 1.0) * rcinv;
}

void CutoffFunction::fdfTANHU(double r, double& fc, double& dfc) const
{
    double const temp = tanh(1.0 - r * rcinv);
    double const temp2 = temp * temp;
    fc = temp * temp2;
    dfc = 3.0 * temp2 * (temp2 - 1.0) * rcinv;
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfTANHU(double r, int order) const
{
    double const x = (1.0 - r * rcinv);
    if(order==0) 
    {
	double t1,t2;
	t1 = tanh(x);
	t2 = t1 * t1;
	return t2 * t1;
    }
    else if(order==1) 
    {
	double t1,t2,t3;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
    	double fact = -rcinv;
	return  fact*(-3.0 * t3 + 3.0 * t2);
    }
    else if(order==2) 
    {
	double t1,t2,t3;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
    	double fact = rcinv*rcinv;
	return fact*(-18.0 * t2 * t1 + 12.0 * t3 * t1 + 6.0 * t1);
    }
    else if(order==3) 
    {
	double t1,t2,t3;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
    	double fact = -rcinv*rcinv*rcinv;
	return fact*(-60 * t3 * t2 - 60 * t2 + 114 * t3 + 6);
    }
    else if(order==4) 
    {
	double t1,t2,t3,t4;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t1;
	t4 = t2 * t2;
    	double fact = rcinv*rcinv*rcinv*rcinv;
	return fact*(-816 * t4 * t1 + 360 * t4 * t3 - 120 * t1 + 576 * t3);
	}
    else if(order==5) 
    {
	double t1,t2,t3,t4;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
	t4 = t3 * t3;
    	double fact = -rcinv*rcinv*rcinv*rcinv*rcinv;
	return fact*(6600 * t3 * t2 + 1848 * t2 - 5808 * t3 - 2520 * t4 - 120);
    }
    else if(order==6) 
    {
	double t1,t2,t3,t4,t5;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
	t4 = t3 * t3;
	t5 = t2 * t1;
    	double fact = rcinv*rcinv*rcinv*rcinv*rcinv*rcinv;
	return fact*(62832 * t3 * t1 + 20160 * t4 * t1 - 59760 * t3 * t5 + 3696 * t1 - 26928 * t5);
    }
    else 
    {
        throw invalid_argument("ERROR: Derivatives over 6 are not yet implemented for thanh.\n");
    }

}
#endif

double CutoffFunction::fTANH(double r) const
{
    double const temp = tanh(1.0 - r * rcinv);
    return TANH_PRE * temp * temp * temp;
}

double CutoffFunction::dfTANH(double r) const
{
    double temp = tanh(1.0 - r * rcinv);
    temp *= temp;
    return 3.0 * TANH_PRE * temp * (temp - 1.0) * rcinv;
}

void CutoffFunction::fdfTANH(double r, double& fc, double& dfc) const
{
    double const temp = tanh(1.0 - r * rcinv);
    double const temp2 = temp * temp;
    fc = TANH_PRE * temp * temp2;
    dfc = 3.0 * TANH_PRE * temp2 * (temp2 - 1.0) * rcinv;
    return;
}

#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfTANH(double r, int order) const
{
    double const x = (1.0 - r * rcinv);
    if(order==0) 
    {
	double t1,t2;
	t1 = tanh(x);
	t2 = t1 * t1;
	return TANH_PRE*t2 * t1;
    }
    else if(order==1) 
    {
	double t1,t2,t3;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
    	double fact = -rcinv*TANH_PRE;
	return  fact*(-3.0 * t3 + 3.0 * t2);
    }
    else if(order==2) 
    {
	double t1,t2,t3;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
    	double fact = rcinv*rcinv*TANH_PRE;
	return fact*(-18.0 * t2 * t1 + 12.0 * t3 * t1 + 6.0 * t1);
    }
    else if(order==3) 
    {
	double t1,t2,t3;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
    	double fact = -rcinv*rcinv*rcinv*TANH_PRE;
	return fact*(-60 * t3 * t2 - 60 * t2 + 114 * t3 + 6);
    }
    else if(order==4) 
    {
	double t1,t2,t3,t4;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t1;
	t4 = t2 * t2;
    	double fact = rcinv*rcinv*rcinv*rcinv*TANH_PRE;
	return fact*(-816 * t4 * t1 + 360 * t4 * t3 - 120 * t1 + 576 * t3);
	}
    else if(order==5) 
    {
	double t1,t2,t3,t4;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
	t4 = t3 * t3;
    	double fact = -rcinv*rcinv*rcinv*rcinv*rcinv*TANH_PRE;
	return fact*(6600 * t3 * t2 + 1848 * t2 - 5808 * t3 - 2520 * t4 - 120);
    }
    else if(order==6) 
    {
	double t1,t2,t3,t4,t5;
	t1 = tanh(x);
	t2 = t1 * t1;
	t3 = t2 * t2;
	t4 = t3 * t3;
	t5 = t2 * t1;
    	double fact = rcinv*rcinv*rcinv*rcinv*rcinv*rcinv*TANH_PRE;
	return fact*(62832 * t3 * t1 + 20160 * t4 * t1 - 59760 * t3 * t5 + 3696 * t1 - 26928 * t5);
    }
    else 
    {
        throw invalid_argument("ERROR: Derivatives over 6 are not yet implemented for thanh.\n");
    }
}
#endif
double CutoffFunction::fEXP(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    return E * exp(1.0 / (x * x - 1.0));
}

double CutoffFunction::dfEXP(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    double const temp = 1.0 / (x * x - 1.0);
    return -2.0 * iw * E * x * temp * temp * exp(temp);
}

void CutoffFunction::fdfEXP(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    double const temp = 1.0 / (x * x - 1.0);
    double const temp2 = exp(temp);
    fc = E * temp2;
    dfc = -2.0 * iw * E * x * temp * temp * temp2;
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfEXP(double r, int order) const
{
    if (r < rci &&  order==0) return 1.0;
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    if(order==0) return E * exp(1.0 / (x * x - 1.0));
    else if(order ==1)
    {
	double t1,t2,t3,t8,t11;
	t1 = x * x;
	t2 = t1 - 0.10e1;
	t3 = t2 * t2;
	t8 = exp(0.10e1 / t2);
	t11 = -0.20e1 * t8 * x / t3;
    	double fact = iw*E;
	return fact*t11;
    }
    else if(order ==2)
    {
	double t1,t2,t3,t10,t17,t18;
	t1 = x * x;
	t2 = t1 - 0.10e1;
	t3 = t2 * t2;
	t10 = t3 * t3;
	t17 = exp(0.10e1 / t2);
	t18 = t17 * (0.80e1 * t1 / t3 / t2 - 0.20e1 / t3 + 0.400e1 * t1 / t10);
    	double fact = iw*iw*E;
	return fact*t18;
    }
    else if(order ==3)
    {
	double t1,t2,t3,t4,t5,t6,t26,t27;
	t1 = x * x;
	t2 = t1 - 0.10e1;
	t3 = t2 * t2;
	t4 = t3 * t3;
	t5 = 0.1e1 / t4;
	t6 = t1 * x;
	t26 = exp(0.10e1 / t2);
	t27 = t26 * (-0.480e2 * t6 * t5 + 0.240e2 * x / t3 / t2 - 0.4800e2 * t6 / t4 / t2 + 0.1200e2 * x * t5 - 0.8000e1 * t6 / t4 / t3);
    	double fact = iw*iw*iw*E;
	return fact*t27;
    }
    else if(order ==4)
    {
	double t1,t2,t3,t4,t6,t7,t10,t14,t17,t29,t36,t37;
	t1 = x * x;
	t2 = t1 - 0.10e1;
	t3 = t2 * t2;
	t4 = t3 * t3;
	t6 = 0.1e1 / t4 / t2;
	t7 = t1 * t1;
	t10 = 0.1e1 / t4;
	t14 = 0.1e1 / t4 / t3;
	t17 = t3 * t2;
	t29 = t4 * t4;
	t36 = exp(0.10e1 / t2);
	t37 = t36 * (0.3840e3 * t7 * t6 - 0.2880e3 * t1 * t10 + 0.57600e3 * t7 * t14 + 0.240e2 / t17 - 0.28800e3 * t1 * t6 + 0.192000e3 * t7 / t4 / t17 + 0.1200e2 * t10 - 0.48000e2 * t1 * t14 + 0.160000e2 * t7 / t29);
    	double fact = iw*iw*iw*iw*E;
	return fact*t37;

    }
    else if(order ==5)
    {
	double t1,t2,t3,t4,t6,t7,t8,t13,t16,t17,t21,t24,t46,t49,t50;
	t1 = x * x;
	t2 = t1 - 0.10e1;
	t3 = t2 * t2;
	t4 = t3 * t3;
	t6 = 0.1e1 / t4 / t3;
	t7 = t1 * t1;
	t8 = t7 * x;
	t13 = 0.1e1 / t4 / t3 / t2;
	t16 = t4 * t4;
	t17 = 0.1e1 / t16;
	t21 = 0.1e1 / t4 / t2;
	t24 = t1 * x;
	t46 = -0.38400e4 * t8 * t6 - 0.768000e4 * t8 * t13 - 0.3840000e4 * t8 * t17 - 0.72000e3 * x * t21 + 0.1920000e4 * t24 * t13 - 0.6400000e3 * t8 / t16 / t2 - 0.120000e3 * x * t6 + 0.1600000e3 * t24 * t17 - 0.3200000e2 * t8 / t16 / t3 + 0.576000e4 * t24 * t6 - 0.7200e3 * x / t4 + 0.38400e4 * t24 * t21;
	t49 = exp(0.10e1 / t2);
	t50 = t49 * t46;
    	double fact = iw*iw*iw*iw*iw*E;
	return fact*t50;

    }
    else if(order ==6)
    {
	double t1,t2,t3,t4,t5,t6,t10,t11,t16,t20,t22,t28,t32,t36,t58,t61,t62;
	t1 = x * x;
	t2 = t1 - 0.10e1;
	t3 = t2 * t2;
	t4 = t3 * t3;
	t5 = t4 * t4;
	t6 = 0.1e1 / t5;
	t10 = 0.1e1 / t5 / t3;
	t11 = t1 * t1;
	t16 = t11 * t1;
	t20 = 0.1e1 / t4 / t3;
	t22 = t3 * t2;
	t28 = 0.1e1 / t5 / t2;
	t32 = 0.1e1 / t4 / t22;
	t36 = 0.1e1 / t4 / t2;
	t58 = 0.7200000e3 * t1 * t6 - 0.48000000e3 * t11 * t10 + 0.64000000e2 * t16 / t5 / t4 - 0.120000e3 * t20 + 0.192000000e4 * t16 / t5 / t22 - 0.96000000e4 * t11 * t28 + 0.8640000e4 * t1 * t32 - 0.72000e3 * t36 + 0.192000000e5 * t16 * t10 + 0.76800000e5 * t16 * t28 + 0.460800e5 * t16 * t32 + 0.11520000e6 * t16 * t6 - 0.576000e5 * t11 * t20 + 0.172800e5 * t1 * t36 - 0.11520000e6 * t11 * t32 - 0.7200e3 / t4 + 0.2592000e5 * t1 * t20 - 0.57600000e5 * t11 * t6;
	t61 = exp(0.10e1 / t2);
	t62 = t61 * t58;
    	double fact = iw*iw*iw*iw*iw*iw*E;
	return fact*t62;
    }
    else 
    {
        throw invalid_argument("ERROR: Derivatives over 6 are not yet implemented for EXP cutoff function.\n");
    }
}
#endif

double CutoffFunction::fPOLY1(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    return (2.0 * x - 3.0) * x * x + 1.0;
}

double CutoffFunction::dfPOLY1(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    return iw * x * (6.0 * x - 6.0);
}

void CutoffFunction::fdfPOLY1(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    fc = (2.0 * x - 3.0) * x * x + 1.0;
    dfc = iw * x * (6.0 * x - 6.0);
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfPOLY1(double r, int order) const
{
    if (r < rci &&  order==0) return 1.0;
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    if(order==0) return (2.0 * x - 3.0) * x * x + 1.0;
    else if(order==1) return iw * x * (6.0 * x - 6.0);
    else if(order==2) return iw*iw*(12 * x - 6);
    else if(order==3) return iw*iw*iw*12;
    else return 0;

}
#endif

double CutoffFunction::fPOLY2(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    return ((15.0 - 6.0 * x) * x - 10.0) * x * x * x + 1.0;
}

double CutoffFunction::dfPOLY2(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    return iw * x * x * ((60.0 - 30.0 * x) * x - 30.0);
}

void CutoffFunction::fdfPOLY2(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    double const x2 = x * x;
    fc = ((15.0 - 6.0 * x) * x - 10.0) * x * x2 + 1.0;
    dfc = iw * x2 * ((60.0 - 30.0 * x) * x - 30.0);
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfPOLY2(double r, int order) const
{
    if (r < rci &&  order==0) return 1.0;
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    if(order==0) return ((15.0 - 6.0 * x) * x - 10.0) * x * x * x + 1.0;
    else if(order==1) return  iw * x * x * ((60.0 - 30.0 * x) * x - 30.0);
    else if(order==2)
    {
	double t1,t2;
	t1 = x * x;
	t2 = -120.0 * t1 * x + 180.0 * t1 - 60.0 * x;
        return iw*iw*t2;
    }
    else if(order==3)
    {
	double t1,t2;
	t1 = x * x;
	t2 = -60.0 - 360.0 * t1 + 360.0 * x;
	return iw*iw*iw*t2;
    }
    else if(order==4)
    {
	return iw*iw*iw*iw*(-720.0 * x + 360.0);
    }
    else if(order==5)
    {
	return -720.0*iw*iw*iw*iw*iw;
    }
    else return 0;
}
#endif

double CutoffFunction::fPOLY3(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    double const x2 = x * x;
    return (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
}

double CutoffFunction::dfPOLY3(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    return iw * x * x * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
}

void CutoffFunction::fdfPOLY3(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    double const x2 = x * x;
    fc = (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
    dfc = iw * x2 * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfPOLY3(double r, int order) const
{
    if (r < rci &&  order==0) return 1.0;
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    if(order==0) return (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x*x*x*x + 1.0;
    else if(order==1) 
    {
	double t1,t2,t3;
	t1 = x * x;
	t2 = t1 * t1;
	t3 = 0.1400e3 * t2 * t1 - 0.4200e3 * t2 * x + 0.4200e3 * t2 - 0.1400e3 * t1 * x;
	return t3*iw;
    }
    else if(order==2) 
    {
	double t1,t2,t3;
	t1 = x * x;
	t2 = t1 * t1;
	t3 = 0.8400e3 * t2 * x - 0.21000e4 * t2 + 0.16800e4 * t1 * x - 0.4200e3 * t1;
	return t3*iw*iw;
    }
    else if(order==3) 
    {
	double t1,t2,t3;
	t1 = x * x;
	t2 = t1 * t1;
	t3 = 0.42000e4 * t2 - 0.84000e4 * t1 * x + 0.50400e4 * t1 - 0.8400e3 * x;
	return t3*iw*iw*iw;
    }
    else if(order==4) 
    {
	double t1,t2;
	t1 = x * x;
	t2 = -0.8400e3 + 0.168000e5 * t1 * x - 0.252000e5 * t1 + 0.100800e5 * x;
	return t2*iw*iw*iw*iw;
    }
    else if(order==5) 
    {
	double t1,t2;
	t1 = x * x;
	t2 = 0.100800e5 + 0.504000e5 * t1 - 0.504000e5 * x;
	return t2*iw*iw*iw*iw*iw;
    }
    else if(order==6) 
    {
	double t2;
	t2 = 0.1008000e6 * x - 0.504000e5;
	return t2*iw*iw*iw*iw*iw*iw;
    }
    else if(order==7) 
    {
	return iw*iw*iw*iw*iw*iw*iw* 0.1008000e6;
    }
    else return 0;
}
#endif

double CutoffFunction::fPOLY4(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    double const x2 = x * x;
    return (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
           x2 * x2 * x + 1.0;
}

double CutoffFunction::dfPOLY4(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    double const x2 = x * x;
    return iw * x2 * x2 *
           (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
}

void CutoffFunction::fdfPOLY4(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    double x4 = x * x;
    x4 *= x4;
    fc = (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
         x * x4 + 1.0;
    dfc = iw * x4 *
          (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
    return;
}
#ifdef HIGH_DERIVATIVES
double CutoffFunction::dnfPOLY4(double r, int order) const
{
    if (r < rci &&  order==0) return 1.0;
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    if(order==0) return (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) * x*x*x*x * x + 1.0;
    else if(order==1) 
    {
	double t1,t2,t3,t4;
	t1 = x * x;
	t2 = t1 * t1;
	t3 = t2 * t2;
	t4 = -0.6300e3 * t3 + 0.25200e4 * t2 * t1 * x - 0.37800e4 * t2 * t1 + 0.25200e4 * t2 * x - 0.6300e3 * t2;
	return t4*iw;
    }
    else if(order==2) 
    {
	double t1,t2,t3,t4;
	t1 = x * x;
	t2 = t1 * x;
	t3 = t1 * t1;
	t4 = -0.50400e4 * t3 * t2 + 0.176400e5 * t3 * t1 - 0.226800e5 * t3 * x + 0.126000e5 * t3 - 0.25200e4 * t2;
	return t4*iw*iw;
    }
    else if(order==3) 
    {
	double t1,t2,t3;
	t1 = x * x;
	t2 = t1 * t1;
	t3 = -0.352800e5 * t2 * t1 + 0.1058400e6 * t2 * x - 0.1134000e6 * t2 + 0.504000e5 * t1 * x - 0.75600e4 * t1;
	return t3*iw*iw*iw;
    }
    else if(order==4) 
    {
	double t1 = x * x;
	double t2 = t1 * t1;
	double t3 = -0.2116800e6 * t2 * x + 0.5292000e6 * t2 - 0.4536000e6 * t1 * x + 0.1512000e6 * t1 - 0.151200e5 * x;
	return t3*iw*iw*iw*iw;
    }
    else if(order==5) 
    {
	double t1 = x * x;
	double t2 = t1 * t1;
	double t3 = -0.151200e5 - 0.10584000e7 * t2 + 0.21168000e7 * t1 * x - 0.13608000e7 * t1 + 0.3024000e6 * x;
	return t3*iw*iw*iw*iw*iw;
    }
    else if(order==6) 
    {
	double t1 = x * x;
	double t2 = 0.3024000e6 - 0.42336000e7 * t1 * x + 0.63504000e7 * t1 - 0.27216000e7 * x;
	return t2*iw*iw*iw*iw*iw*iw;
    }
    else if(order==7) 
    {
	double t1 = x * x;
	double t2 = -0.27216000e7 - 0.127008000e8 * t1 + 0.127008000e8 * x;
	return iw*iw*iw*iw*iw*iw*iw*t2; 
    }
    else if(order==8) 
    {
	double t2 = -0.254016000e8 * x + 0.127008000e8;
	return iw*iw*iw*iw*iw*iw*iw*iw*t2; 
    }
    else if(order==9) 
    {
	return -iw*iw*iw*iw*iw*iw*iw*iw*iw*0.254016000e8;
    }
    else return 0;
}
#endif
#ifdef HIGH_DERIVATIVES
/* firstDeriv vector with 6 elements */
/* rij[3] : xi,yi,zi,xj,yj,zj*/
double CutoffFunction::getFirstDerivatives(const double* rij, double firstDeriv[]) const
{
	double r = getDerivativesRij(rij,  firstDeriv); 
	double dfdr= dnf(r, 1);
	for(size_t l=0;l<6;l++) firstDeriv[l] *= dfdr;
	return dnf(r, 0);
	
}
/* firstDeriv vector with 6 elements */
/* secondDeriv matrix with 6x6 elements */
double CutoffFunction::get12Derivatives(const double* rij,  double firstDeriv[], double secondDeriv[][6]) const
{
	double dr[6];
	double d2r[6][6];
	double r = getDerivativesRij(rij,  dr, d2r); 
	double dfdr= dnf(r, 1);
	double d2fdr2= dnf(r, 2);
	for(size_t l=0;l<6;l++) firstDeriv[l] = dr[l]*dfdr;
	for(size_t l=0;l<6;l++)   
	for(size_t m=l;m<6;m++)   secondDeriv[l][m] = secondDeriv[m][l] = dr[l]*dr[m]*d2fdr2 + d2r[l][m]*dfdr;

	return dnf(r,0);
}
#endif
