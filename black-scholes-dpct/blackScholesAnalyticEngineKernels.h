//blackScholesAnalyticEngineKernels.cuh
//Scott Grauer-Gray
//Kernels for running black scholes using the analytic engine

#ifndef BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_CUH
#define BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>

//needed for the structs used on the code
#include "blackScholesAnalyticEngineStructs.h"

//constants used in this code
#define M_1_SQRTPI  0.564189583547756286948
#define M_SQRT_2    0.7071067811865475244008443621048490392848359376887


//device kernel to retrieve the compound factor in interestRate
 float interestRateCompoundFactor(float t, yieldTermStruct currYieldTermStruct);

//device kernel to retrieve the discount factor in interestRate
 float interestRateDiscountFactor(float t, yieldTermStruct currYieldTermStruct);


//device function to get the variance of the black volatility function
 float getBlackVolBlackVar(blackVolStruct volTS);


//device function to get the discount on a dividend yield
 float getDiscountOnDividendYield(float yearFraction, yieldTermStruct dividendYieldTermStruct);


//device function to get the discount on the risk free rate
 float getDiscountOnRiskFreeRate(float yearFraction, yieldTermStruct riskFreeRateYieldTermStruct);

//device kernel to run the error function
 float errorFunct(normalDistStruct normDist, float x);


//device kernel to run the operator function in cumulative normal distribution
 float cumNormDistOp(normalDistStruct normDist, float z);


//device kernel to run the gaussian function in the normal distribution
 float gaussianFunctNormDist(normalDistStruct normDist, float x);

//device kernel to retrieve the derivative in a cumulative normal distribution
 float cumNormDistDeriv(normalDistStruct normDist, float x);

//device function to initialize the cumulative normal distribution structure
 void initCumNormDist(normalDistStruct* currCumNormDist);


//device function to initialize variable in the black calculator
 void initBlackCalcVars(blackCalcStruct* blackCalculator,  payoffStruct payoff);


//device function to initialize the black calculator
 void initBlackCalculator(blackCalcStruct* blackCalc, payoffStruct payoff, float forwardPrice, float stdDev, float riskFreeDiscount);


//device function to retrieve the output resulting value
 float getResultVal(blackCalcStruct* blackCalculator);


//global function to retrieve the output value for an option
void getOutValOption(optionInputStruct* options, float* outputVals, int numVals);

#endif //BLACK_SCHOLES_ANALYTIC_ENGINE_KERNELS_CUH
