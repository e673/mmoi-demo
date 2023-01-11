#pragma once

#include "../core.h"

namespace ip
{
	class Metrics
	{
	public:
		static float PSNR(const ip::ImageFloat& img1, const ip::ImageFloat& img2);
		static float SSIM(const ip::ImageFloat& img1, const ip::ImageFloat& img2, float sigma);
		static void SSIM(const ip::ImageFloat& img1, const ip::ImageFloat& img2, float sigma, ip::Imageloat& dst);
	};
}