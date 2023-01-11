#include "../../iplib/image/metrics/metrics.h"
#include <immintrin.h>
#include <iplib/parallel.h>
#include <math.h>
#include <iplib/image/filter/filter.hpp>

namespace ip
{
	float Metrics::PSNR(const ip::ImageFloat& img1, const ip::ImageFloat& img2)
	{
		double mse = ip::Parallel::For<double>(0, img1.Height(), [&img1, &img2](int y, double& state)
		{
			float sum = 0.0f;

			for (int x = 0; x < img1.Width(); x++)
			{
				float q = img1(x, y) - img2(x, y);
				sum += q * q;
			}

			state += sum;
		},
		[](double& x, const double& y) { x += y; });

		return (float)(10.0 * log10(255.0 * 255.0 * img1.Width() * img1.Height() / mse));
	}

	static void Mul(const ip::ImageFloat& src1, const ip::ImageFloat &src2, ip::ImageFloat &dst)
	{
		ip::Parallel::For(0, src1.Height(), [&src1, &src2, &dst](int y)
		{
			for (int x = 0; x < src1.Width(); x++)
			{
				dst(x, y) = src1(x, y) * src2(x, y);
			}
		});
	}

	float Metrics::SSIM(const ip::ImageFloat& img1, const ip::ImageFloat& img2, float sigma)
	{
		ImageFloat tmp(img1.Width(), img2.Height());
		SSIM(img1, img2, sigma, tmp);

		double aver = 0.0;

		for (int j = 0; j < img1.Height(); j++)
			for (int i = 0; i < img1.Width(); i++)
				aver += tmp(i, j);

		return (float)aver / (img1.Width() * img1.Height());
	}

	void Metrics::SSIM(const ip::ImageFloat& img1, const ip::ImageFloat& img2, float sigma, ip::ImageFloat& dst)
	{	
		ImageFloat m1(img1.Width(), img1.Height());
		Filter::Gauss(img1, m1, sigma);

		ImageFloat m2(img2.Width(), img2.Height());
		Filter::Gauss(img2, m2, sigma);

		ImageFloat tmp(img1.Width(), img1.Height());
		Mul(img1, img1, tmp);

		ImageFloat sigma11(img1.Width(), img1.Height());
		Filter::Gauss(tmp, sigma11, sigma);

		for (int j = 0; j < img1.Height(); j++)
			for (int i = 0; i < img1.Width(); i++)
				sigma11(i, j) -= m1(i, j) * m1(i, j);

		Mul(img1, img2, tmp);

		ImageFloat sigma12(img1.Width(), img1.Height());
		Filter::Gauss(tmp, sigma12, sigma);

		for (int j = 0; j < img1.Height(); j++)
			for (int i = 0; i < img1.Width(); i++)
				sigma12(i, j) -= m1(i, j) * m2(i, j);

		Mul(img2, img2, tmp);

		ImageFloat sigma22(img1.Width(), img1.Height());
		Filter::Gauss(tmp, sigma22, sigma);

		for (int j = 0; j < img1.Height(); j++)
			for (int i = 0; i < img1.Width(); i++)
				sigma22(i, j) -= m2(i, j) * m2(i, j);

		float c1 = 0.01f * 255.0f;
		float c2 = 0.03f * 255.0f;
		c1 *= c1;
		c2 *= c2;

		for (int j = 0; j < img1.Height(); j++)
			for (int i = 0; i < img1.Width(); i++)
				dst(i, j) = ((2.0f * m1(i, j) * m2(i, j) + c1) * (2.0f * sigma12(i, j) + c2) / ((m1(i, j) * m1(i, j) + m2(i, j) * m2(i, j) + c1) * (sigma11(i, j) + sigma22(i, j) + c2)));
	}
}