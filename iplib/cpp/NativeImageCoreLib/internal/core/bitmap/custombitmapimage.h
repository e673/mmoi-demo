#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "bitmapimage.h"

namespace ip
{
	template <typename PixelType>
	class CustomBitmapImage
		: public BitmapImage <PixelType, false>
	{
	public:
		CustomBitmapImage()
		{
			this->data.data = 0;
			this->data.width = 0;
			this->data.height = 0;
			this->data.stride = 0;
		}

		void Init(void *data, int Width, int Height, ptrdiff_t Stride)
		{
			this->data.data = data;
			this->data.width = Width;
			this->data.height = Height;
			this->data.stride = Stride;
		}
	};
}