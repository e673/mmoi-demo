#pragma once

#include "../core.h"

namespace ip
{
	class VarMethods
	{
	public:
		static void Convolve(const Image<float> &x, const Image<float> &kernel, Image<float> &dst);

		static void DiscrepancyL1(const Image<float> &z, const Image<float> &u, const Image<float> &kernel, Image<float> &dst, Image<float> &tmp);
		static void DiscrepancyL2(const Image<float> &z, const Image<float> &u, const Image<float> &kernel, Image<float> &dst, Image<float> &tmp);

		static float CalcNormL1(const Image<float> &x);
		static float CalcNormL2(const Image<float> &x);

		static void NormalizeGradientL1(Image<float> &x, float target_norm);

		// L1 gradient norm
		static void AddGradientTVL1(const Image<float> &z, Image<float> &dst, float alpha);

		// L2 gradient, forward derivatives
		static void AddGradientForwardTVL2(const Image<float> &z, Image<float> &dst, float alpha);

		// L2 gradient, central derivatives
		static void AddGradientCentralTVL2(const Image<float> &z, Image<float> &dst, float alpha);

		// BTV gradient
		static void AddGradientBTV(const Image<float> &z, Image<float> &dst, float alpha);

		// Second order BTV gradient
		static void AddGradientBTV2(const Image<float> &z, Image<float> &dst, float alpha);

		static void Subtract(Image<float> &dst, const Image<float> &src1, const Image<float> &src2, float q);
		static void ApplyMomentum(Image<float> &dst, const Image<float> &cur, Image<float> &prev, float mu);
	};
}