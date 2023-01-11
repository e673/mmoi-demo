#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include <memory>

#include "../base/imagewritable3d.h"
#include "bitmapdatastructure.h"
#include "../../../iplib/common.h"

namespace ip
{
	// ==================================================================================================
	//                                           BitmapImage3D
	// ==================================================================================================

	template <typename PixelType, bool Aligned>
	class BitmapImage3D
		: public ImageWritable3D < PixelType, BitmapImage3D<PixelType, Aligned> >
	{
	protected:
		BitmapDataStructure3D data;
		std::shared_ptr<internal::BitmapDataResource<BitmapDataStructure3D>> base_data;

	public:
		inline int SizeX() const
		{
			return data.size_x;
		}

		inline int SizeY() const
		{
			return data.size_y;
		}

		inline int SizeZ() const
		{
			return data.size_z;
		}

		inline const PixelType* pixeladdr(int x, int y, int z) const
		{
			dbgcheck(x >= 0 && y >= 0 && z >= 0 && x < data.size_x && y < data.size_y && z < data.size_z);
			return (PixelType*)((char*)data.data + y * data.stride_y + z * data.stride_z) + x;
		}

		inline PixelType* pixeladdr(int x, int y, int z)
		{
			dbgcheck(x >= 0 && y >= 0 && z >= 0 && x < data.size_x && y < data.size_y && z < data.size_z);
			return (PixelType*)((char*)data.data + y * data.stride_y + z * data.stride_z) + x;
		}

		inline PixelType pixel(int x, int y, int z) const
		{
			return *pixeladdr(x, y, z);
		}

		inline PixelType& pixel(int x, int y, int z)
		{
			return *pixeladdr(x, y, z);
		}
	};
}