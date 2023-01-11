#pragma once

#include <iplib/image/core.h>
#include <memory>

namespace ip
{
	class SRCNN_Resampling;

	class SRCNN
	{
	public:
		SRCNN(wchar_t *filename = nullptr);
		operator bool() const;
		bool Resample_x2_915(const ip::Image<float> &src, ip::Image<float> &dst, bool half_shift);
		~SRCNN();

	private:
		SRCNN_Resampling *impl;
	};
}