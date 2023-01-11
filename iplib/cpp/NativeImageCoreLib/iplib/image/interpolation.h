#pragma once

#include "core.h"

namespace ip
{
	template <typename PixelType, class ImageType>
	PixelType Bilinear(const ImageReadable<PixelType, ImageType> &image, float x, float y)
	{
		int x0 = (int)x;
		if (x0 < 0)
		{
			x0 = 0;
			x = 0.0f;
		}
		else if (x0 > image.Width() - 2)
		{
			x0 = image.Width() - 2;
			x = 1.0f;
		}
		else
		{
			x -= x0;
		}

		int y0 = (int)y;
		if (y0 < 0)
		{
			y0 = 0;
			y = 0.0f;
		}
		else if (y0 > image.Height() - 2)
		{
			y0 = image.Height() - 2;
			y = 1.0f;
		}
		else
		{
			y -= y0;
		}

		return (image(x0, y0) * (1.0f - x) + image(x0 + 1, y0) * x) * (1.0f - y) +
			(image(x0, y0 + 1) * (1.0f - x) + image(x0 + 1, y0 + 1) * x) * y;
	}

	template <typename PixelType, class ImageType>
	PixelType Bicubic(const ImageReadable<PixelType, ImageType> &image, float x, float y)
	{
		int x0 = (int)x;
		if (x0 < 1 || x0 > image.Width() - 2)
			return Bilinear(image, x, y);

		int y0 = (int)y;
		if (y0 < 1 || y0 >= image.Height() - 2)
			return Bilinear(image, x, y);

		x -= x0;
		y -= y0;

		float x2 = x * x;
		float x3 = x2 * x;

		float y2 = y * y;
		float y3 = y2 * y;

		struct helper
		{
			PixelType a0, a1, a2, a3;

			inline helper(const ImageReadable<PixelType, ImageType> &image, int x, int y)
			{
				a0 = image(x - 1, y);
				a1 = image(x, y);
				a2 = image(x + 1, y);
				a3 = image(x + 2, y);
			}

			inline helper(PixelType a0, PixelType a1, PixelType a2, PixelType a3)
				: a0(a0), a1(a1), a2(a2), a3(a3) {}

			inline PixelType operator()(float x, float x2, float x3)
			{
				return ((2.0f * a1) + x * (a2 - a0) + x2 * (2.0f * a0 - 5.0f * a1 + 4.0f * a2 - a3) +
					x3 * ((a1 - a2) * 3.0f + a3 - a0)) * 0.5f;
			}
		};

		PixelType a0 = helper(image, x0, y0 - 1)(x, x2, x3);
		PixelType a1 = helper(image, x0, y0)(x, x2, x3);
		PixelType a2 = helper(image, x0, y0 + 1)(x, x2, x3);
		PixelType a3 = helper(image, x0, y0 + 2)(x, x2, x3);

		return helper(a0, a1, a2, a3)(y, y2, y3);
	}
}