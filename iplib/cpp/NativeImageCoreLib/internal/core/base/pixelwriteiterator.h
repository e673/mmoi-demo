#pragma once

#ifndef IPLIB_INCLUDE_BASE_H
#error This file should not be included directly
#endif

#include "imagebasefwd.h"

namespace ip
{
	// ==================================================================================================
	//                                   PixelWriteIterator - interface              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	class PixelWriteIterator
	{
	public:
		// Create default pixel write iterator
		PixelWriteIterator(ImageWritable<PixelType, ImageType>& image, int x, int y);

		// Set the pixel value with offset N from the current position
		void set(PixelType value, int N = 0);

		// Advance the current position by N pixels
		void advance(int N);

	private:
		ImageWritable<PixelType, ImageType>& image;
		int x, y;
	};

	// ==================================================================================================
	//                                   PixelWriteIterator - implementation              
	// ==================================================================================================

	template <typename PixelType, class ImageType, typename Enable>
	PixelWriteIterator<PixelType, ImageType, Enable>::PixelWriteIterator(ImageWritable<PixelType, ImageType>& image, int x, int y)
		: image(image), x(x), y(y) {}

	template <typename PixelType, class ImageType, typename Enable>
	void PixelWriteIterator<PixelType, ImageType, Enable>::set(PixelType value, int PixelOffset)
	{
		image.pixel(x + PixelOffset, y) = value;
	}

	template <typename PixelType, class ImageType, typename Enable>
	void PixelWriteIterator<PixelType, ImageType, Enable>::advance(int PixelOffset)
	{
		x += PixelOffset;
	}
}