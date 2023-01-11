#include "../../iplib/image/resampling/edresampling.h"

#include <math.h>

namespace ip
{
	// ========================================================================================================

	void EDResampling::Kernels::LoadDefault()
	{
		kernel0[0] = -0.00063f;
		kernel0[1] = -0.00180f;
		kernel0[2] = 0.02694f;
		kernel0[3] = 0.01716f;
		kernel0[4] = -0.15097f;
		kernel0[5] = 1.45306f;

		kernel1[0] = 0.03297f;
		kernel1[1] = -0.04665f;
		kernel1[2] = -0.04484f;
		kernel1[3] = -0.05382f;
		kernel1[4] = 0.12650f;
		kernel1[5] = 0.58139f;

		kernel2[0] = -0.03016f;
		kernel2[1] = -0.04952f;
		kernel2[2] = -0.05838f;
		kernel2[3] = 0.04283f;
		kernel2[4] = 0.45204f;
		kernel2[5] = 0.25095f;
	}

	// ========================================================================================================

	EDResampling::EDResampling()
	{
		kernels.LoadDefault();
	}

	EDResampling::EDResampling(EDResampling::Kernels kernels)
	{
		this->kernels = kernels;
	}

	void EDResampling::Configure(bool fast_weights)
	{
		this->fast_weights = fast_weights;
	}

	void EDResampling::Average3x3(const Image<float>& src, Image<float>& dst)
	{
		Parallel::For(1, src.Height() - 1, [&src, &dst](int j)
		{
			for (int i = 1; i < src.Width() - 1; i++)
			{
				dst(i, j) = src(i - 1, j - 1) + src(i, j - 1) + src(i + 1, j - 1) +
					src(i - 1, j) + src(i, j) + src(i + 1, j) +
					src(i - 1, j + 1) + src(i, j + 1) + src(i + 1, j + 1);
			}
		});
	}

	void EDResampling::Average3x3diag(const Image<float>& src, Image<float>& dst)
	{
		Parallel::For(3, src.Height() - 3, [&src, &dst](int j)
		{
			for (int i = 3 + (j % 2); i < src.Width() - 3; i += 2)
			{
				dst(i, j) = src(i, j - 2) + src(i + 1, j - 1) + src(i + 2, j) +
					src(i - 1, j - 1) + src(i, j) + src(i + 1, j + 1) +
					src(i - 2, j) + src(i - 1, j + 1) + src(i, j + 2);
			}
		});
	}

	float EDResampling::CalcWeight(float g1, float g2)
	{
		/* if ((1.0f + g1) > (1.0f + g2) * Threshold)
			return 0.0f;

		if ((1.0f + g2) > (1.0f + g1) * Threshold)
			return 1.0f; */

		float G1 = g1 * g1 * g1;
		float G2 = g2 * g2 * g2;

		float GG1 = G1 * G1;
		float GG2 = G2 * G2;

		return (1.0f + GG2) / (2.0f + GG1 + GG2);
	}

	float EDResampling::CalcWeightFast(float g1, float g2)
	{
		if (g1 > g2 * Threshold)
			return 0.0f;
		else if (g2 > g1 * Threshold)
			return 1.0f;
		else
			return 0.5f;
	}

	// ========================================================================================================

	void EDResamplingExt::UpdateCoefficients()
	{
		UpdateCoefficientsStep1();
		UpdateCoefficientsStep2();
	}

	void EDResamplingExt::UpdateCoefficientsStep1()
	{
		opt0.CalcOptimizedCoefficients(kernels.kernel0);
		opt1.CalcOptimizedCoefficients(kernels.kernel1);
	}

	void EDResamplingExt::UpdateCoefficientsStep2()
	{
		opt2.CalcOptimizedCoefficients(kernels.kernel2);
		opt2a.CalcOptimizedCoefficients(kernels.kernel2a);
		opt2b.CalcOptimizedCoefficients(kernels.kernel2b);
		opt2c.CalcOptimizedCoefficients(kernels.kernel2c);
	}

	EDResampling::Kernels EDResamplingExt::GetKernels() const
	{
		return kernels;
	}

	// ========================================================================================================

	EDResampling::WeightCalculator::WeightCalculator(bool fast, int Width, int Height)
		: data(Width, Height), fast(fast) {}

	float EDResampling::WeightCalculator::get(int x, int y) const
	{
		return data(x, y);
	}
}