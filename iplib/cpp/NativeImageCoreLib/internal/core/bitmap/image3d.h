#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "bitmapimage3d.h"
#include "../base/pixeltypes.h"

namespace ip
{
	namespace internal
	{
		class BitmapData3D
			: public BitmapDataResource<BitmapDataStructure3D>
		{
		public:
			BitmapData3D(int SizeX, int SizeY, int SizeZ, int PixelSize);
			virtual ~BitmapData3D();

			static const int Alignment = 32;
		};
	}

	// ==========================================================================================

	template <typename PixelType>
	class Image3D
		: public BitmapImage3D < PixelType, true >
	{
	public:
		Image3D()
		{
			this->data.data = nullptr;
			this->data.size_x = 0;
			this->data.size_y = 0;
			this->data.size_z = 0;
			this->data.stride_y = 0;
			this->data.stride_z = 0;
		}

		Image3D(int SizeX, int SizeY, int SizeZ)
		{
			this->base_data.reset(new internal::BitmapData3D(SizeX, SizeY, SizeZ, sizeof(PixelType)));
			this->data = *this->base_data;
		}

		Image3D(Image3D<PixelType> &&move_from)
		{
			swap(std::move(move_from));
		}

		void swap(Image3D<PixelType> &&other)
		{
			this->data = other.data;
			this->base_data.swap(other.base_data);
			other.data.data = nullptr;
			other.data.size_x = other.data.size_y = other.data.size_z = 0;
			other.data.stride_y = other.data.stride_z = 0;
		}
	};

	// ==========================================================================================
}