#include "../../iplib/math/gauss_function.h"
#include "../../iplib/math/quadratic_optimization.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace ip
{
	// ==================================================================================================

	GaussFunction::GaussFunction(float sigma, bool normalize)
		: qx(-1.0f / (2.0f * sigma * sigma))
	{
		if (normalize)
			q = 1.0f / (sqrtf(2.0f * (float)M_PI) * sigma);
		else
			q = 1.0f;
	}

	float GaussFunction::operator()(float x) const
	{
		return q * expf(x * x * qx);
	}

	float GaussFunction::calc(float sigma, float x)
	{
		return expf(x * x / (-2.0f * sigma * sigma)) / (sqrtf(2.0f * (float)M_PI) * sigma);
	}

	// ==================================================================================================

	GaussFunctionDerivative::GaussFunctionDerivative(float sigma, bool normalize)
		: qx(-1.0f / (2.0f * sigma * sigma))
	{
		if (normalize)
			q = -1.0f / (sqrtf(2.0f * (float)M_PI) * sigma * sigma * sigma);
		else
			q = -1.0f;
	}

	float GaussFunctionDerivative::operator()(float x) const
	{
		return q * x * expf(x * x * qx);
	}

	float GaussFunctionDerivative::calc(float sigma, float x)
	{
		return -expf(x * x / (-2.0f * sigma * sigma)) * x / (sqrtf(2.0f * (float)M_PI) * sigma * sigma * sigma);
	}

	// ==================================================================================================

	GaussFunctionSecondDerivative::GaussFunctionSecondDerivative(float sigma, bool normalize)
		: qx(-1.0f / (2.0f * sigma * sigma))
	{
		if (normalize)
			q = -2.0f / (sqrtf(2.0f * (float)M_PI) * sigma * sigma * sigma);
		else
			q = -2.0f;
	}

	float GaussFunctionSecondDerivative::operator()(float x) const
	{
		float y = x * x * qx;
		return q * expf(y) * (0.5f + y);
	}

	float GaussFunctionSecondDerivative::calc(float sigma, float x)
	{
		float q = -2.0f / (sqrtf(2.0f * (float)M_PI) * sigma * sigma * sigma);
		float y = -x * x / (2.0f * sigma * sigma);
		return q * expf(y) * (0.5f + y);
	}

	// ==================================================================================================
	//                                   QuadraticOptimization               
	// ==================================================================================================

	void QuadraticOptimization::Reset()
	{
		A = Matrix<double>();
		B = Matrix<double>();
		N = 0;
	}

	int QuadraticOptimization::Count() const
	{
		return N;
	}

	int QuadraticOptimization::GetVectorLength() const
	{
		return A.NumColumns();
	}
}