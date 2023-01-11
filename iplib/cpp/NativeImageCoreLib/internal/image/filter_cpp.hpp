#include "../../iplib/image/filter/filter.hpp"
#include <iplib/parallel.h>
#include <algorithm>

namespace ip
{
	void Filter::FilterHorizontal(const ip::ImageFloat& src, float* kernel, int kernel_center, int kernel_length, ip::ImageFloat& dst)
	{
		ip::Parallel::For(0, src.Height(), [&src, kernel, kernel_center, kernel_length, &dst](int y)
		{
			int x = 0;

			for (; x < kernel_center; x++)
			{
				float p = 0.0f;
				for (int k = 0; k < kernel_length; k++)
				{
					int x0 = (std::max)(x + k - kernel_center, 0);
					p += src(x0, y) * kernel[k];
				}
				dst(x, y) = p;
			}

			int n = src.Width() + kernel_center - kernel_length - 7;

			for (; x <= n; x += 8)
			{
				__m256 p = _mm256_setzero_ps();

				for (int k = 0; k < kernel_length; k++)
				{
					p = _mm256_add_ps(p, _mm256_mul_ps(_mm256_loadu_ps(src.pixeladdr(x + k - kernel_center, y)), _mm256_broadcast_ss(kernel + k)));
				}

				_mm256_storeu_ps(dst.pixeladdr(x, y), p);
			}

			for (; x < src.Width(); x++)
			{
				float p = 0.0f;
				for (int k = 0; k < kernel_length; k++)
				{
					int x0 = (std::min)(x + k - kernel_center, src.Width() - 1);
					p += src(x0, y) * kernel[k];
				}
				dst(x, y) = p;
			}
		});
	}

	void Filter::FilterVertical(const ip::ImageFloat& src, float* kernel, int kernel_center, int kernel_length, ip::ImageFloat& dst)
	{
		ip::Parallel::For(0, src.Height(), [&src, kernel, kernel_center, kernel_length, &dst](int y)
		{
			if (y < kernel_center || y > src.Height() + kernel_center - kernel_length)
			{
				for (int x = 0; x < src.Width(); x++)
				{
					float p = 0.0f;

					for (int k = 0; k < kernel_length; k++)
					{
						int y0 = (std::min)((std::max)(y + k - kernel_center, 0), src.Height() - 1);
						p += src(x, y0) * kernel[k];
					}

					dst(x, y) = p;
				}
			}
			else
			{
				int x = 0;

				for (; x < src.Width() - 7; x += 8)
				{
					__m256 p = _mm256_setzero_ps();

					for (int k = 0; k < kernel_length; k++)
					{
						p = _mm256_add_ps(p, _mm256_mul_ps(_mm256_loadu_ps(src.pixeladdr(x, y + k - kernel_center)), _mm256_broadcast_ss(kernel + k)));
					}

					_mm256_storeu_ps(dst.pixeladdr(x, y), p);

				}

				for (; x < src.Width(); x++)
				{
					float p = 0.0f;

					for (int k = 0; k < kernel_length; k++)
					{
						p += src(x, y + k - kernel_center) * kernel[k];
					}

					dst(x, y) = p;
				}
			}
		});
	}

	void Filter::Filter2D(const ip::ImageFloat& src, const ip::ImageFloat& kernel, int cx, int cy, ip::ImageFloat& dst)
	{
		ip::Parallel::For(0, src.Height(), [&src, &kernel, cx, cy, &dst](int y)
		{
			for (int x = 0; x < src.Width(); x++)
			{
				float p = 0.0f;

				for (int j = 0; j < kernel.Height(); j++)
					for (int i = 0; i < kernel.Width(); i++)
					{
						int x0 = (std::min)((std::max)(x + i - cx, 0), src.Width() - 1);
						int y0 = (std::min)((std::max)(y + j - cy, 0), src.Height() - 1);
						p += src(x0, y0) * kernel(i, j);
					}

				dst(x, y) = p;
			}
		});
	}

	void Filter::Gauss(const ip::ImageFloat& src, ip::ImageFloat& dst, float sigma)
	{
		ImageFloat tmp(src.Width(), src.Height());
		Gauss(src, tmp, dst, sigma);
	}

	void Filter::Gauss(const ip::ImageFloat& src, ip::ImageFloat& tmp, ip::ImageFloat& dst, float sigma)
	{
		int hsize = (int)ceilf(3.0f * sigma);
		std::vector<float> filter(hsize * 2 + 1);

		filter[hsize] = 1.0f;

		for (int i = 1; i <= hsize; i++)
		{
			filter[hsize + i] = filter[hsize - i] = expf(i * i / (-2.0f * sigma * sigma));
		}

		float sum = 0.0f;
		for (int i = 0; i <= 2 * hsize; i++)
			sum += filter[i];

		for (int i = 0; i <= 2 * hsize; i++)
			filter[i] /= sum;

		FilterHorizontal(src, filter.data(), hsize, 2 * hsize + 1, tmp);
		FilterVertical(tmp, filter.data(), hsize, 2 * hsize + 1, dst);
	}
}
