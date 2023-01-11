#pragma once

#include <iplib/image/core.h>
#include <iplib/image/canny.h>
#include <iplib/parallel.h>
#include <iplib/image/motion.h>

#define _USE_MATH_DEFINES
#include <math.h>

namespace ip
{
	class MeowWarping
	{
	public:
		MeowWarping();

		template <typename SourcePixelType, class SourceImageType, class DestinationImageType, class DisplacementFunctionType>
		void GenerateWarping2D(const ImageReadable<SourcePixelType, SourceImageType> &src,
			ImageWritable<PixelFloatVector, DestinationImageType> &dst,
			DisplacementFunctionType &df);

		template <typename SourcePixelType, class SourceImageType>
		Image<SourcePixelType> Warp2D(const ImageReadable<SourcePixelType, SourceImageType> &src);

		template <typename SourcePixelType, class SourceImageType>
		Image<SourcePixelType> Warp2D_2(const ImageReadable<SourcePixelType, SourceImageType> &src);


	private:
		float edge_detector_sigma;
		float warping_sigma;
		float warping_power;
		float normal_weight_sigma_q;
		float tangent_weight_sigma_q;
		float interpolation_sigma;
		Image<float> warping_sigma_image;

		template <class DisplacementFunctionType>
		void ApplyFunc(int x0, int y0, Image<PixelFloatVector> &q, ImageFloat &q0, PixelFloatVector grad, DisplacementFunctionType &df);

	public:
		void SetEdgeDetectionSigma(float sigma) { edge_detector_sigma = sigma; }
		void SetWarpingSigma(float sigma) { warping_sigma = sigma; }
		void SetWarpingPower(float power) { warping_power = power; }
		
		void SetWarpingSigmaImage(const Image<float> &img)
		{
			warping_sigma_image.swap(Image<float>(img.Width(), img.Height()));
			img.CopyTo(warping_sigma_image);
		}
	};

	MeowWarping::MeowWarping()
		: edge_detector_sigma(2.0f)
		, warping_sigma(2.0f)
		, warping_power(1.0f)
		, normal_weight_sigma_q(100.0f)
		, tangent_weight_sigma_q(2.5f)
		, interpolation_sigma(0.3f) {}
	

	template <typename SourcePixelType, class SourceImageType, class DestinationImageType, class DisplacementFunctionType>
	void MeowWarping::GenerateWarping2D(const ImageReadable<SourcePixelType, SourceImageType> &src, ImageWritable<PixelFloatVector, DestinationImageType> &dst, DisplacementFunctionType &df)
	{
		// printf("Canny edge detection...");

		Image<float> gs(src.Width(), src.Height());
		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				gs(i, j) = (float)src(i, j);

		Canny canny(gs, edge_detector_sigma);
		Image<PixelFloatVector> grad = canny.GetGradient();

		// printf(" done\nFinding warping vectors...");

		Image<PixelFloatVector> q(src.Width(), src.Height());
		ImageFloat q0(src.Width(), src.Height());

		for (int j = 0; j < src.Height(); j++)
		{
			for (int i = 0; i < src.Width(); i++)
			{
				q(i, j) = PixelFloatVector();
				q0(i, j) = 0.0f;
			}
		}

		if (warping_sigma_image)
		{
			for (int j = 0; j < src.Height(); j++)
			{
				for (int i = 0; i < src.Width(); i++)
				{
					if (canny(i, j) > 0.0f)
						ApplyFunc(i, j, q, q0, grad(i, j), df);

					q0(i, j) += 1e-6f;
				}
			}
		}
		else
		{
			for (int j = 0; j < src.Height(); j++)
			{
				for (int i = 0; i < src.Width(); i++)
				{
					if (canny(i, j) > 0.0f)
						ApplyFunc(i, j, q, q0, grad(i, j), df);

					q0(i, j) += 1e-6f;
				}
			}
		}

		for (int j = 0; j < src.Height(); j++)
			for (int i = 0; i < src.Width(); i++)
				dst(i, j) = q(i, j) * (1.0f / q0(i, j));

		// printf(" done\n");
	}

	template <class DisplacementFunctionType>
	void MeowWarping::ApplyFunc(int x0, int y0, Image<PixelFloatVector> &q, ImageFloat &q0, PixelFloatVector grad, DisplacementFunctionType &df)
	{
		int rad = (int)(warping_sigma * 5.0f);
		float d = grad.Norm();
		grad.x /= d;
		grad.y /= d;

		float xr = -1.0f / (2.0f * warping_sigma * warping_sigma * normal_weight_sigma_q * normal_weight_sigma_q);
		float yr = -1.0f / (2.0f * warping_sigma * warping_sigma * tangent_weight_sigma_q * tangent_weight_sigma_q);

		Parallel::For(-rad, rad + 1, [rad, x0, y0, xr, yr, &q, &q0, grad, &df, d](int j)
		{
			for (int i = -rad; i <= rad; i++)
			{
				int tx = x0 + i;
				int ty = y0 + j;
				if (tx < 0 || tx >= q.Width() || ty < 0 || ty >= q.Height())
					continue;

				float x = i * grad.x + j * grad.y;
				float y = -i * grad.y + j * grad.x;
				float ws = d * expf(x * x * xr) * expf(y * y * yr);

				PixelFloatVector v = grad * df(x);

				q(tx, ty) += v * ws;
				q0(tx, ty) += ws;
			}
		});
	}

	template <typename SourcePixelType, class SourceImageType>
	Image<SourcePixelType> MeowWarping::Warp2D(const ImageReadable<SourcePixelType, SourceImageType> &src)
	{
		constexpr float k = 1.0f / 2.0f;
		float ws = 1.0f / warping_sigma;
		float q = (warping_sigma * sqrtf(2.0f * (float)M_PI)) / (1.0f - k) * 0.5f;

		Image<PixelFloatVector> vectors(src.Width(), src.Height());

		GenerateWarping2D(src, vectors, [q, ws, k](float x) -> float
		{
			return q * (erff(x * ws * k) - erff(x * ws));
		});

		Image<SourcePixelType> res(src.Width(), src.Height());
		ForwardMotionWarping(src, vectors, res, this->interpolation_sigma, this->warping_power);
		return res;
	}

	template <typename SourcePixelType, class SourceImageType>
	Image<SourcePixelType> MeowWarping::Warp2D_2(const ImageReadable<SourcePixelType, SourceImageType> &src)
	{
		float ws = warping_sigma;
		float nws = 1.0f / warping_sigma;

		Image<PixelFloatVector> vectors(src.Width(), src.Height());

		GenerateWarping2D(src, vectors, [ws, nws](float x) -> float
		{
			float y = x * nws;
			float d = y < 1.0f ? -y : (y < 2.0f ? y - 2.0f : 0.0f);
			return d * ws;
		});

		Image<SourcePixelType> res(src.Width(), src.Height());
		ForwardMotionWarping(src, vectors, res, this->interpolation_sigma, this->warping_power);
		return res;
	}
}
