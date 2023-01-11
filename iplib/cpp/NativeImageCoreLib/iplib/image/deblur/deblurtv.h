#pragma once

#include "../core.h"
#include "../core3d.h"
#include <iplib/parallel.h>

namespace ip
{
	class DeblurTV
	{
	public:
		static void AnyKernel(const Image<float> &src, Image<float> &dst, const Image<float> &kernel, float reg_param);
		static void AnyKernel2(const Image<float> &src, Image<float> &dst, const Image<float> &kernel, float reg_param_1, float reg_param_2, int num_iter, float mu, float corr_factor);
		static void AnyKernel3(const Image3D<float> &src, Image3D<float> &dst, const Image3D<float> &kernel, float reg_param_1, float reg_param_2, int num_iter, float mu);
	};
}