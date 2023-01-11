#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include <memory>

#include "bitmapdatastructure.h"
#include "../base/imagewritable.h"
#include "../../../iplib/common.h"

namespace ip
{
	// ==================================================================================================
	//                                           BitmapImage               
	// ==================================================================================================

	template <typename PixelType, bool Aligned>
	class BitmapImage
		: public ImageWritable < PixelType, BitmapImage<PixelType, Aligned> >
	{
	protected:
		BitmapDataStructure data;
		std::shared_ptr<internal::BitmapDataResource<BitmapDataStructure>> base_data;

	public:
		const BitmapDataStructure& Data() const
		{
			return data;
		}

		operator bool () const
		{
			return data.data != nullptr;
		}

		int Width() const
		{
			return data.width;
		}

		int Height() const
		{
			return data.height;
		}

		const PixelType* pixeladdr(int x, int y) const
		{
			dbgcheck(x >= 0 && y >= 0 && x < data.width && y < data.height);
			return (PixelType*)((char*)data.data + y * data.stride) + x;
		}

		PixelType* pixeladdr(int x, int y)
		{
			dbgcheck(x >= 0 && y >= 0 && x < data.width && y < data.height);
			return (PixelType*)((char*)data.data + y * data.stride) + x;
		}

		PixelType pixel(int x, int y) const
		{
			return *pixeladdr(x, y);
		}

		PixelType& pixel(int x, int y)
		{
			return *pixeladdr(x, y);
		}
	};
}