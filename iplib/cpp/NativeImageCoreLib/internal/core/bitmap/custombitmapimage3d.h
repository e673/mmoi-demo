#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "bitmapimage3d.h"

namespace ip
{
	template <typename PixelType>
	class CustomBitmapImage3D
		: public BitmapImage3D <PixelType, false>
	{
	public:
		CustomBitmapImage3D()
		{
			this->data.data = 0;
			this->data.size_x = 0;
			this->data.size_y = 0;
			this->data.size_z = 0;
			this->data.stride_y = 0;
			this->data.stride_z = 0;
		}

		void Init(void *data, int SizeX, int SizeY, int SizeZ, ptrdiff_t StrideY, ptrdiff_t StrideZ)
		{
			this->data.data = data;
			this->data.size_x = SizeX;
			this->data.size_y = SizeY;
			this->data.size_z = SizeZ;
			this->data.stride_y = StrideY;
			this->data.stride_z = StrideZ;
		}
	};
}