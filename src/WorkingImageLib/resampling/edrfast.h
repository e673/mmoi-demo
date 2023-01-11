#pragma once

#include <iplib/image/core.h>

namespace ip
{
	bool EDR_Resampling_x2(const ip::Image<float> &src, ip::Image<float> &dst);
	bool EDR_Resampling_x2(const ip::ImageFloatColor &src, ip::ImageFloatColor &dst);
}