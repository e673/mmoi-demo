#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

namespace ip
{
	// ==================================================================================================
	//                                   PixelReadIterator3D - interface
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class PixelReadIterator3D
	{
	public:
		// Create default pixel read iterator
		PixelReadIterator3D(const ImageBase3D<PixelType, ImageType>& image, int x, int y, int z);

		// Get the pixel value with offset N from the current position
		PixelType get(int PixelOffset = 0);

		// Advance the current position by N pixels
		void advance(int PixelOffset);

	private:
		const ImageBase3D<PixelType, ImageType>& image;
		int x, y, z;
	};

	// ==================================================================================================
	//                                 PixelReadIterator3D - implementation
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	PixelReadIterator3D<PixelType, ImageType, Enable>::PixelReadIterator3D(const ImageBase3D<PixelType, ImageType>& image, int x, int y, int z)
		: image(image), x(x), y(y), z(z) {}

	template <typename PixelType, class ImageType, typename Enable>
	PixelType PixelReadIterator3D<PixelType, ImageType, Enable>::get(int PixelOffset)
	{
		return image.pixel(x + PixelOffset, y, z);
	}

	template <typename PixelType, class ImageType, typename Enable>
	void PixelReadIterator3D<PixelType, ImageType, Enable>::advance(int PixelOffset)
	{
		x += PixelOffset;
	}
}