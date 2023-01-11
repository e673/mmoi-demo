#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

namespace ip
{
	// ==================================================================================================
	//                                 PixelWriteIterator3D - interface              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class PixelWriteIterator3D
	{
	public:
		// Create default pixel write iterator
		PixelWriteIterator3D(ImageWritable3D<PixelType, ImageType>& image, int x, int y, int z);

		// Set the pixel value with offset N from the current position
		void set(PixelType value, int N = 0);

		// Advance the current position by N pixels
		void advance(int N);

	private:
		ImageWritable3D<PixelType, ImageType>& image;
		int x, y, z;
	};

	// ==================================================================================================
	//                                PixelWriteIterator3D - implementation              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	PixelWriteIterator3D<PixelType, ImageType, Enable>::PixelWriteIterator3D(ImageWritable3D<PixelType, ImageType>& image, int x, int y, int z)
		: image(image), x(x), y(y), z(z) {}

	template <typename PixelType, class ImageType, typename Enable>
	void PixelWriteIterator3D<PixelType, ImageType, Enable>::set(PixelType value, int PixelOffset)
	{
		image.pixel(x + PixelOffset, y, z) = value;
	}

	template <typename PixelType, class ImageType, typename Enable>
	void PixelWriteIterator3D<PixelType, ImageType, Enable>::advance(int PixelOffset)
	{
		x += PixelOffset;
	}
}