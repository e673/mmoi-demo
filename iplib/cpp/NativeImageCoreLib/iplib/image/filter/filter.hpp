#pragma once

#include "../core.h"

namespace ip
{
	class Filter
	{
	public:
		static void FilterHorizontal(const ip::ImageFloat& src, float* kernel, int kernel_center, int kernel_length, ip::ImageFloat& dst);
		static void FilterVertical(const ip::ImageFloat& src, float* kernel, int kernel_center, int kernel_length, ip::ImageFloat& dst);
		static void Filter2D(const ip::ImageFloat& src, const ip::ImageFloat& kernel, int cx, int cy, ip::ImageFloat& dst);
		static void Gauss(const ip::ImageFloat& src, ip::ImageFloat& dst, float sigma);
		static void Gauss(const ip::ImageFloat& src, ip::ImageFloat& tmp, ip::ImageFloat& dst, float sigma);
	};
}
