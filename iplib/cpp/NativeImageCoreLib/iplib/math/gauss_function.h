#pragma once

namespace ip
{
	class GaussFunction
	{
		float qx, q;

	public:
		GaussFunction(float sigma, bool normalize = true);
		float operator ()(float x) const;
		static float calc(float sigma, float x);
	};

	class GaussFunctionDerivative
	{
		float qx, q;

	public:
		GaussFunctionDerivative(float sigma, bool normalize = true);
		float operator ()(float x) const;
		static float calc(float sigma, float x);
	};

	class GaussFunctionSecondDerivative
	{
		float qx, q;

	public:
		GaussFunctionSecondDerivative(float sigma, bool normalize = true);
		float operator ()(float x) const;
		static float calc(float sigma, float x);
	};
}
